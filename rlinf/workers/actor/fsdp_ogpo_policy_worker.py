# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-dimensional OGPO worker built on RLinf's embodied SAC infrastructure.

The worker preserves OGPO's two-stage recipe for the Relocate benchmark:
flow-matching behavior cloning followed by online TD critic learning and a
group-relative, Q-guided clipped policy update with BC regularization.
"""

from __future__ import annotations

import math
import os
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.ogpo_io_struct import OGPOTrajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _OGPOSingleGPUStrategyAdapter:
    """Use plain modules for single-GPU OGPO and delegate every other operation."""

    def __init__(self, strategy, *, device: torch.device, dtype: torch.dtype):
        self._strategy = strategy
        self._device = device
        self._dtype = dtype

    def __getattr__(self, name: str):
        return getattr(self._strategy, name)

    def wrap_model(self, model: torch.nn.Module, device_mesh) -> torch.nn.Module:
        del device_mesh
        return model.to(device=self._device, dtype=self._dtype)


def _use_ogpo_single_gpu_fast_path(cfg: DictConfig, world_size: int) -> bool:
    """Select the strictly scoped plain-module fast path."""
    return (
        bool(cfg.actor.get("ogpo_single_gpu_fast_path", False))
        and world_size == 1
        and str(cfg.actor.fsdp_config.get("strategy", "fsdp")).lower() == "fsdp"
        and str(cfg.actor.fsdp_config.sharding_strategy).lower() == "no_shard"
        and not bool(cfg.actor.fsdp_config.get("cpu_offload", False))
        and not bool(cfg.actor.get("enable_offload", False))
    )


def _get_ogpo_online_micro_batch_size(cfg: DictConfig, world_size: int) -> int:
    """Use the full OGPO batch only on the isolated single-GPU fast path."""
    micro_batch_size = int(cfg.actor.micro_batch_size)
    if _use_ogpo_single_gpu_fast_path(cfg, world_size):
        micro_batch_size = int(
            cfg.actor.get("ogpo_single_gpu_micro_batch_size", micro_batch_size)
        )
    per_rank_batch_size = int(cfg.actor.global_batch_size) // world_size
    if micro_batch_size <= 0 or per_rank_batch_size % micro_batch_size != 0:
        raise ValueError(
            "OGPO micro-batch size must be positive and divide the per-rank "
            f"batch size ({per_rank_batch_size}), got {micro_batch_size}."
        )
    return micro_batch_size


def _get_pretrained_target_actor_state(converted: dict) -> dict | None:
    """Prefer a saved EMA actor and fall back to an actor-only BC checkpoint."""
    target_actor = converted.get("target_actor")
    if target_actor is not None:
        return target_actor
    return converted.get("actor")


def _repeat_batch(value: Any, repeats: int) -> Any:
    """Repeat a nested observation batch contiguously for batched QVR."""
    if torch.is_tensor(value):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_batch(item, repeats) for key, item in value.items()}
    raise TypeError(f"Unsupported batched QVR observation type: {type(value)!r}")


def compute_discounted_chunk_return(
    rewards: torch.Tensor, gamma: float, horizon: int
) -> torch.Tensor:
    """Return the discounted reward accumulated by one action chunk."""
    chunk_rewards = rewards[:, :horizon]
    powers = torch.arange(
        horizon, device=chunk_rewards.device, dtype=chunk_rewards.dtype
    )
    return (chunk_rewards * (gamma**powers).unsqueeze(0)).sum(dim=-1, keepdim=True)


class _ActionChunkDataset(Dataset):
    """View a flat transition dataset as arbitrary-start fixed-length chunks."""

    def __init__(self, dataset: Dataset, horizon: int):
        self.dataset = dataset
        self.horizon = horizon
        self.dones = torch.as_tensor(dataset.dones_float, dtype=torch.bool)
        explicit_starts = getattr(dataset, "action_chunk_start_indices", None)
        if explicit_starts is None:
            self.valid_starts = torch.arange(
                max(0, len(dataset) - horizon + 1), dtype=torch.long
            )
        else:
            self.valid_starts = torch.as_tensor(explicit_starts, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.valid_starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.valid_starts[index])
        end = start + self.horizon
        dones = self.dones[start:end]
        valid = torch.ones(self.horizon, dtype=torch.float32)
        if self.horizon > 1:
            valid[1:] = (~torch.cummax(dones[:-1], dim=0).values).float()
        return {
            "observations": torch.from_numpy(self.dataset.observations[start]).float(),
            "actions": torch.from_numpy(self.dataset.actions[start:end])
            .float()
            .flatten(),
            "valid": valid,
        }


def compute_group_relative_advantages(
    q_values: torch.Tensor,
    aggregation: str = "mean",
    normalize: bool = False,
    strategy: str = "vanilla",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute OGPO group-relative advantages from a Q ensemble.

    Args:
        q_values: Tensor shaped ``[group, batch, num_q]``.
        aggregation: Ensemble aggregation, either ``mean`` or ``min``.
        normalize: Whether to divide each group's centered values by its
            standard deviation.
        strategy: ``vanilla`` mean-centering or OGPO's ``conservative``
            sign-consensus aggregation across Q heads.

    Returns:
        A pair ``(advantages, aggregated_q)``, both shaped ``[group, batch]``.
    """
    if q_values.ndim != 3:
        raise ValueError(
            f"q_values must have shape [group, batch, num_q], got {q_values.shape}"
        )
    if aggregation == "mean":
        aggregated_q = q_values.mean(dim=-1)
    elif aggregation == "min":
        aggregated_q = q_values.min(dim=-1).values
    else:
        raise ValueError(f"Unsupported OGPO Q aggregation: {aggregation!r}")

    if strategy == "vanilla":
        advantages = aggregated_q - aggregated_q.mean(dim=0, keepdim=True)
        if normalize:
            advantages = advantages / (
                advantages.std(dim=0, keepdim=True, unbiased=False) + 1e-8
            )
    elif strategy == "conservative":
        per_q_advantages = q_values - q_values.mean(dim=0, keepdim=True)
        minimum = per_q_advantages.min(dim=-1).values
        maximum = per_q_advantages.max(dim=-1).values
        advantages = minimum * (minimum > 0) + maximum * (maximum < 0)
    else:
        raise ValueError(f"Unsupported OGPO advantage strategy: {strategy!r}")
    return advantages, aggregated_q


class EmbodiedOGPOFSDPPolicy(EmbodiedSACFSDPPolicy):
    """RLinf-native OGPO worker for state-based continuous-control tasks."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.offline_data_loader: DataLoader | None = None
        self.offline_data_iter: Any = None
        self._offline_sampler: DistributedSampler | None = None
        self._offline_epoch = 0
        self._online_phase_initialized = False
        self._force_bc_finished = False
        self._bc_update_steps = 0
        self._online_env_steps = 0
        self._primitive_windows: dict[
            tuple[int, int], deque[dict[str, torch.Tensor]]
        ] = {}
        self._pending_replay_sequences: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._episode_ids: dict[tuple[int, int], int] = {}
        self._episode_success: dict[tuple[int, int], bool] = {}
        self._episode_success_labels: dict[tuple[int, int, int], bool] = {}
        self._success_bc_samples: deque[dict[str, torch.Tensor]] = deque(
            maxlen=int(self.cfg.algorithm.get("success_buffer_size", 2_000_000))
        )
        self._using_success_bc = False
        self._ogpo_rollout_weight_source = "target"
        self._ogpo_profile_traces = 0

    def init_worker(self) -> None:
        """Initialize OGPO and enable its optional single-GPU TF32 policy."""
        if torch.cuda.is_available() and bool(self.cfg.actor.get("allow_tf32", False)):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
        if _use_ogpo_single_gpu_fast_path(self.cfg, self._world_size):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dtype = torch_dtype_from_precision(
                self.cfg.actor.fsdp_config.mixed_precision.param_dtype
            )
            self._strategy = _OGPOSingleGPUStrategyAdapter(
                self._strategy,
                device=torch.device(f"{Worker.torch_device_type}:{local_rank}"),
                dtype=dtype,
            )
            self.log_info(
                "OGPO single-GPU fast path: using a plain module instead of "
                "FSDP NO_SHARD."
            )
        super().init_worker()
        if bool(self.cfg.algorithm.get("q_vr_batched", False)):
            qvr_batch = int(self.cfg.actor.global_batch_size) * int(
                self.cfg.algorithm.get("q_vr_num_samples", 8)
            )
            self.model.q_head.vectorized_batch_limit = max(
                self.model.q_head.vectorized_batch_limit, qvr_batch
            )
            self.target_model.q_head.vectorized_batch_limit = max(
                self.target_model.q_head.vectorized_batch_limit, qvr_batch
            )
        if bool(self.cfg.actor.get("ogpo_compile_critic", False)):
            compile_mode = str(
                self.cfg.actor.get("ogpo_compile_critic_mode", "default")
            )
            # nn.Module.compile() changes __call__ in place, so optimizer
            # parameters and checkpoint state-dict keys remain unchanged.
            self.model.q_head.compile(mode=compile_mode)
            self.target_model.q_head.compile(mode=compile_mode)
            self.log_info(f"Compiled OGPO online and target critic ({compile_mode=}).")
        converted_path = self.cfg.actor.model.get("pretrained_actor_path")
        if converted_path:
            converted = torch.load(
                converted_path, map_location="cpu", weights_only=False
            )
            target_actor = _get_pretrained_target_actor_state(converted)
            if target_actor is not None:
                target_state = self._strategy.get_model_state_dict(
                    self.target_model, cpu_offload=True, full_state_dict=True
                )
                for name, value in target_actor.items():
                    target_state[f"flow_actor.{name}"] = value
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    target_state,
                    cpu_offload=True,
                    full_state_dict=True,
                )

    def set_ogpo_rollout_weight_source(self, source: str) -> None:
        """Select current weights for ODE or EMA target weights for SDE."""
        normalized = str(source).lower()
        if normalized not in {"current", "target"}:
            raise ValueError(f"Unsupported OGPO rollout weight source: {source!r}")
        self._ogpo_rollout_weight_source = normalized

    def get_rollout_state_dict(self) -> dict:
        """Export the official target policy/Q for stochastic online rollout."""
        if self._ogpo_rollout_weight_source == "current":
            return super().get_rollout_state_dict()
        return self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=False
        )

    def setup_model_and_optimizer(self, initialize_target: bool = False) -> None:
        """Use the official AdamW optimizers for OGPO actor and critic."""
        super().setup_model_and_optimizer(initialize_target=initialize_target)

        def rebuild(optimizer, optim_cfg):
            params = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            fused = optim_cfg.get("fused")
            if fused is None:
                fused = torch.cuda.is_available() and all(
                    parameter.is_cuda for parameter in params
                )
            return torch.optim.AdamW(
                params,
                lr=float(optim_cfg.lr),
                betas=(
                    float(optim_cfg.get("adam_beta1", 0.9)),
                    float(optim_cfg.get("adam_beta2", 0.999)),
                ),
                eps=float(optim_cfg.get("adam_eps", 1e-8)),
                weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
                fused=bool(fused),
            )

        self.optimizer = rebuild(self.optimizer, self.cfg.actor.optim)
        self.qf_optimizer = rebuild(self.qf_optimizer, self.cfg.actor.critic_optim)
        self.build_lr_schedulers()
        self._cache_ogpo_parameter_partitions()

    def setup_sac_components(self) -> None:
        """Initialize online replay and the offline expert dataloader."""
        super().setup_sac_components()
        if bool(self.cfg.runner.get("only_eval", False)):
            self.log_info("OGPO evaluation-only run: skipping the offline dataset.")
            return
        dataset_type = str(self.cfg.data.get("dataset_type", "d4rl")).lower()
        if dataset_type == "realworld_franka_paligemma":
            from rlinf.data.datasets.realworld_franka_paligemma import (
                build_realworld_franka_paligemma_dataset_from_cfg,
            )

            dataset = build_realworld_franka_paligemma_dataset_from_cfg(self.cfg)
        elif dataset_type == "robomimic":
            from rlinf.data.datasets.robomimic import (
                build_robomimic_dataset_from_cfg,
            )

            dataset = build_robomimic_dataset_from_cfg(self.cfg)
        else:
            from rlinf.data.datasets.d4rl import build_d4rl_dataset_from_cfg

            dataset = build_d4rl_dataset_from_cfg(self.cfg)
        horizon = int(self.cfg.actor.model.get("num_action_chunks", 1))
        dataset = _ActionChunkDataset(dataset, horizon=horizon)
        batch_size = int(self.cfg.actor.global_batch_size) // self._world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._offline_sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=bool(self.cfg.data.get("shuffle", True)),
                seed=int(self.cfg.data.get("seed", self.cfg.actor.seed)),
                drop_last=True,
            )
        self.offline_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._offline_sampler,
            shuffle=self._offline_sampler is None,
            num_workers=int(self.cfg.data.get("num_workers", 0)),
            drop_last=True,
            pin_memory=bool(self.cfg.data.get("pin_memory", True)),
        )
        self.offline_data_iter = iter(self.offline_data_loader)
        self.log_info(
            "OGPO offline expert dataset: "
            f"{len(dataset)} transitions, per-rank batch size {batch_size}."
        )

    def _clear_online_sequence_state(self) -> None:
        self._primitive_windows.clear()
        self._pending_replay_sequences.clear()
        self._episode_ids.clear()
        self._episode_success.clear()
        self._episode_success_labels.clear()
        self._success_bc_samples.clear()
        self._using_success_bc = False

    def _success_buffer_min_samples(self) -> int:
        """Mirror official batch_size * UTD + horizon readiness threshold."""
        global_batch = int(
            self.cfg.algorithm.get(
                "success_buffer_batch_size", self.cfg.actor.global_batch_size
            )
        )
        utd = int(self.cfg.algorithm.get("success_buffer_utd", 4))
        horizon = int(self.cfg.actor.model.get("num_action_chunks", 1))
        return math.ceil((global_batch * utd + horizon) / self._world_size)

    def _build_sliding_sequence_trajectory(
        self, trajectory: Trajectory
    ) -> Trajectory | None:
        """Commit completed episodes as official arbitrary-start H-step sequences."""
        required = (
            trajectory.primitive_curr_states,
            trajectory.primitive_next_states,
            trajectory.primitive_actions,
            trajectory.primitive_rewards,
            trajectory.primitive_successes,
            trajectory.primitive_terminations,
            trajectory.primitive_truncations,
        )
        if any(value is None for value in required):
            raise ValueError(
                "OGPO online replay requires primitive transition tensors from EnvWorker."
            )
        horizon = int(self.cfg.actor.model.get("num_action_chunks", 1))
        source_rank = int(trajectory.source_rank)
        primitive_curr_states = trajectory.primitive_curr_states
        completed_samples: list[dict[str, Any]] = []
        for time_id in range(primitive_curr_states.shape[0]):
            for env_id in range(primitive_curr_states.shape[1]):
                key = (source_rank, env_id)
                window = self._primitive_windows.setdefault(key, deque(maxlen=horizon))
                pending = self._pending_replay_sequences.setdefault(key, [])
                episode_id = self._episode_ids.setdefault(key, 0)
                for primitive_id in range(primitive_curr_states.shape[2]):
                    transition = {
                        "curr_state": trajectory.primitive_curr_states[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "next_state": trajectory.primitive_next_states[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "action": trajectory.primitive_actions[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "reward": trajectory.primitive_rewards[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "success": trajectory.primitive_successes[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "termination": trajectory.primitive_terminations[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "truncation": trajectory.primitive_truncations[
                            time_id, env_id, primitive_id
                        ].clone(),
                        "episode_id": episode_id,
                    }
                    window.append(transition)
                    if len(window) == horizon:
                        sequence = list(window)
                        terminations = torch.stack(
                            [item["termination"] for item in sequence]
                        )
                        truncations = torch.stack(
                            [item["truncation"] for item in sequence]
                        )
                        dones = torch.logical_or(terminations, truncations).bool()
                        valid = torch.ones(horizon, dtype=torch.float32)
                        if horizon > 1:
                            valid[1:] = (
                                ~torch.cummax(dones[:-1], dim=0).values
                            ).float()
                        pending.append(
                            {
                                "curr_state": sequence[0]["curr_state"],
                                "next_state": sequence[-1]["next_state"],
                                "actions": torch.stack(
                                    [item["action"] for item in sequence]
                                ).flatten(),
                                "rewards": torch.stack(
                                    [item["reward"] for item in sequence]
                                ),
                                "terminations": terminations,
                                "truncations": truncations,
                                "valid": valid,
                                "start_episode_id": sequence[0]["episode_id"],
                            }
                        )
                    done = bool(transition["termination"] or transition["truncation"])
                    if done:
                        label_key = (source_rank, env_id, episode_id)
                        # Official OGPO labels the whole trajectory using only
                        # the success value on its terminal frame.
                        self._episode_success_labels[label_key] = bool(
                            transition["success"]
                        )
                        for sample in pending:
                            completed_samples.append(sample)
                            start_label_key = (
                                source_rank,
                                env_id,
                                int(sample["start_episode_id"]),
                            )
                            if self._episode_success_labels.get(start_label_key, False):
                                self._success_bc_samples.append(
                                    {
                                        "observations": sample["curr_state"].clone(),
                                        "next_observations": sample[
                                            "next_state"
                                        ].clone(),
                                        "actions": sample["actions"].clone(),
                                        "rewards": sample["rewards"].clone(),
                                        "terminations": sample["terminations"].clone(),
                                        "truncations": sample["truncations"].clone(),
                                        "valid": sample["valid"].clone(),
                                    }
                                )
                        pending.clear()
                        episode_id += 1
                        self._episode_ids[key] = episode_id
                        self._episode_success[key] = False
                        self._episode_success_labels.pop(
                            (source_rank, env_id, episode_id - 2), None
                        )
        if not completed_samples:
            return None
        terminations = torch.stack(
            [sample["terminations"] for sample in completed_samples]
        )
        truncations = torch.stack(
            [sample["truncations"] for sample in completed_samples]
        )
        return OGPOTrajectory(
            source_rank=source_rank,
            actions=torch.stack(
                [sample["actions"] for sample in completed_samples]
            ).unsqueeze(1),
            rewards=torch.stack(
                [sample["rewards"] for sample in completed_samples]
            ).unsqueeze(1),
            valid=torch.stack(
                [sample["valid"] for sample in completed_samples]
            ).unsqueeze(1),
            terminations=terminations.unsqueeze(1),
            truncations=truncations.unsqueeze(1),
            dones=torch.logical_or(terminations, truncations).unsqueeze(1),
            curr_obs={
                "states": torch.stack(
                    [sample["curr_state"] for sample in completed_samples]
                ).unsqueeze(1)
            },
            next_obs={
                "states": torch.stack(
                    [sample["next_state"] for sample in completed_samples]
                ).unsqueeze(1)
            },
        )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive rollouts and add official arbitrary-start sequences online only."""
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        sequence_trajectories = []
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            if not self._online_phase_initialized:
                continue
            sequence_trajectory = self._build_sliding_sequence_trajectory(trajectory)
            if sequence_trajectory is not None:
                sequence_trajectories.append(sequence_trajectory)
        if sequence_trajectories:
            self.replay_buffer.add_trajectories(sequence_trajectories)

    def _online_bc_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Use successful online episodes, falling back to the replay minibatch."""
        batch_size = batch["actions"].shape[0]
        self._using_success_bc = (
            len(self._success_bc_samples) >= self._success_buffer_min_samples()
        )
        if self._using_success_bc:
            indices = torch.randint(len(self._success_bc_samples), (batch_size,))
            samples = [self._success_bc_samples[int(index)] for index in indices]
            return {
                "observations": torch.stack(
                    [sample["observations"] for sample in samples]
                ).to(self.device, non_blocking=True),
                "actions": torch.stack([sample["actions"] for sample in samples]).to(
                    self.device, non_blocking=True
                ),
                "valid": torch.stack([sample["valid"] for sample in samples]).to(
                    self.device, non_blocking=True
                ),
            }
        actions = batch["actions"]
        dones = torch.logical_or(batch["terminations"], batch["truncations"]).bool()
        valid = torch.ones_like(dones, dtype=torch.float32)
        if valid.shape[-1] > 1:
            valid[:, 1:] = (~torch.cummax(dones[:, :-1], dim=1).values).float()
        return {
            "observations": batch["curr_obs"]["states"],
            "actions": actions.reshape(actions.shape[0], -1),
            "valid": valid,
        }

    def _online_success_q_batch(self, batch_size: int) -> dict[str, Any] | None:
        """Sample an official-format TD batch from successful trajectories."""
        if len(self._success_bc_samples) < self._success_buffer_min_samples():
            return None
        indices = torch.randint(len(self._success_bc_samples), (batch_size,))
        samples = [self._success_bc_samples[int(index)] for index in indices]

        def stack(name: str) -> torch.Tensor:
            return torch.stack([sample[name] for sample in samples]).to(
                self.device, non_blocking=True
            )

        return {
            "curr_obs": {"states": stack("observations")},
            "next_obs": {"states": stack("next_observations")},
            "actions": stack("actions"),
            "rewards": stack("rewards"),
            "terminations": stack("terminations"),
            "truncations": stack("truncations"),
            "valid": stack("valid"),
        }

    def _next_offline_batch(self) -> dict[str, torch.Tensor]:
        assert self.offline_data_loader is not None
        try:
            batch = next(self.offline_data_iter)
        except StopIteration:
            self._offline_epoch += 1
            if self._offline_sampler is not None:
                self._offline_sampler.set_epoch(self._offline_epoch)
            self.offline_data_iter = iter(self.offline_data_loader)
            batch = next(self.offline_data_iter)
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _flow_bc_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = {"states": batch["observations"].to(self.torch_dtype)}
        actions = batch["actions"].to(self.torch_dtype)
        predicted_velocity, target_velocity = self.model(
            forward_type=ForwardType.OGPO_BC,
            obs=obs,
            actions=actions,
        )
        squared_error = (predicted_velocity - target_velocity).square()
        valid = batch.get("valid")
        if valid is None:
            return squared_error.mean()
        horizon = int(self.cfg.actor.model.get("num_action_chunks", 1))
        action_dim = int(self.cfg.actor.model.action_dim)
        mask = valid.to(squared_error.dtype).reshape(-1, horizon, 1)
        squared_error = squared_error.reshape(-1, horizon, action_dim)
        return (squared_error * mask).sum() / (mask.sum() * action_dim).clamp_min(1.0)

    def _run_bc_training(self) -> dict[str, float]:
        updates = int(self.cfg.algorithm.get("bc_updates_per_step", 1))
        losses: list[torch.Tensor] = []
        grad_norms: list[torch.Tensor] = []
        self.model.train()
        for _ in range(updates):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._flow_bc_loss(self._next_offline_batch())
            loss.backward()
            grad_norm = self._clip_ogpo_grad_norm(
                actor=True, max_norm=float(self.cfg.actor.optim.clip_grad)
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            # Official OGPO updates the target actor after every BC update.
            # The shared state policy also contains an unchanged Q head, so
            # applying the EMA to the whole module is equivalent here.
            self._soft_update_ogpo_target(actor=True)
            losses.append(loss.detach())
            grad_norms.append(torch.as_tensor(grad_norm).detach())

        # Convert once per runner step instead of forcing two host/device
        # synchronizations for every BC update.
        mean_loss = torch.stack(losses).mean().item()
        mean_grad_norm = torch.stack(grad_norms).mean().item()
        self._bc_update_steps += updates

        metrics = {
            "ogpo/phase": 0.0,
            "ogpo/bc_loss": mean_loss,
            "actor/grad_norm": mean_grad_norm,
            "actor/lr": float(self.optimizer.param_groups[0]["lr"]),
            "ogpo/bc_updates": float(updates),
            "ogpo/bc_update_steps": float(self._bc_update_steps),
            "ogpo/online_env_steps": 0.0,
        }
        return all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)

    def finish_ogpo_bc(self) -> None:
        """Switch phases before the first online rollout is inserted."""
        self._force_bc_finished = True
        self._enter_online_phase()

    def _reset_online_optimizers(self) -> None:
        """Reset AdamW state while preserving official actor and critic LRs."""
        actor_lr = float(self.cfg.algorithm.get("online_actor_lr", 4.5e-5))
        critic_lr = float(self.cfg.actor.critic_optim.lr)
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            group["lr"] = actor_lr
            group["initial_lr"] = actor_lr
        self.qf_optimizer.state.clear()
        for group in self.qf_optimizer.param_groups:
            group["lr"] = critic_lr
            group["initial_lr"] = critic_lr
        actor_optim_cfg = OmegaConf.create(
            OmegaConf.to_container(self.cfg.actor.optim, resolve=True)
        )
        actor_optim_cfg.lr_scheduler = self.cfg.algorithm.get(
            "online_actor_lr_scheduler", actor_optim_cfg.lr_scheduler
        )
        actor_optim_cfg.lr_warmup_steps = int(
            self.cfg.algorithm.get("online_actor_lr_warmup_steps", 0)
        )
        actor_optim_cfg.total_training_steps = int(
            self.cfg.algorithm.get("online_actor_total_training_steps", 0)
        )
        actor_optim_cfg.min_lr = float(
            self.cfg.algorithm.get("online_actor_min_lr", 0.0)
        )
        self.lr_scheduler = self.build_lr_scheduler(self.optimizer, actor_optim_cfg)
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def restore_ogpo_training_state(
        self, bc_update_steps: int, online_env_steps: int, bc_finished: bool
    ) -> None:
        """Restore phase counters and make a transition checkpoint resumable."""
        self._bc_update_steps = int(bc_update_steps)
        self._online_env_steps = int(online_env_steps)
        self._force_bc_finished = bool(bc_finished)
        if not bc_finished:
            return

        if online_env_steps == 0:
            # A transition checkpoint is written before the first online
            # rollout. Discard stale BC-time rollouts from legacy checkpoints.
            self.replay_buffer.clear()
            self.buffer_dataloader_iter = iter(self.buffer_dataloader)
            self._clear_online_sequence_state()
            self._reset_online_optimizers()
        self._online_phase_initialized = True

    def _enter_online_phase(self) -> None:
        if self._online_phase_initialized:
            return
        # Rollouts generated while BC was running are not part of OGPO's
        # online replay distribution. Start online learning from a clean buffer.
        self.replay_buffer.clear()
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)
        self._clear_online_sequence_state()
        self._reset_online_optimizers()
        online_lr = float(self.cfg.algorithm.get("online_actor_lr", 4.5e-5))
        self._online_phase_initialized = True
        self.log_info(
            "OGPO BC phase complete; cleared warmup replay and switched actor "
            f"learning rate to {online_lr:g}."
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch, collect_metrics: bool = True):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_n_step_target = bool(
            self.cfg.algorithm.get("use_action_chunk_n_step_target", False)
        )
        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        elif use_n_step_target:
            num_action_chunks = int(self.cfg.actor.model.get("num_action_chunks", 1))
            chunk_rewards = batch["rewards"].to(self.torch_dtype)
            rewards_for_bootstrap = compute_discounted_chunk_return(
                chunk_rewards, self.cfg.algorithm.gamma, num_action_chunks
            )
            discount = self.cfg.algorithm.gamma**num_action_chunks
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            bootstrap_actor = (
                self.target_model
                if bool(self.cfg.algorithm.get("use_target_actor_for_bootstrap", False))
                else self.model
            )
            next_state_actions, next_state_log_pi, shared_feature = bootstrap_actor(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                if bool(self.cfg.algorithm.get("q_variance_reduction", False)):
                    num_samples = int(self.cfg.algorithm.get("q_vr_num_samples", 8))
                    reduction = str(self.cfg.algorithm.get("q_vr_reduction", "mean"))
                    if num_samples < 1:
                        raise ValueError("q_vr_num_samples must be positive")
                    if bool(self.cfg.algorithm.get("q_vr_batched", False)):
                        batch_size = next_state_actions.shape[0]
                        repeated_obs = _repeat_batch(next_obs, num_samples)
                        action_shape = next_state_actions.shape[1:]
                        if num_samples > 1:
                            additional_obs = _repeat_batch(next_obs, num_samples - 1)
                            additional_actions, _, _ = bootstrap_actor(
                                forward_type=ForwardType.SAC,
                                obs=additional_obs,
                                **kwargs,
                            )
                            sampled_actions = torch.cat(
                                (
                                    next_state_actions.unsqueeze(1),
                                    additional_actions.reshape(
                                        batch_size, num_samples - 1, *action_shape
                                    ),
                                ),
                                dim=1,
                            ).reshape(batch_size * num_samples, *action_shape)
                        else:
                            sampled_actions = next_state_actions
                        sampled_q_values = self.target_model(
                            forward_type=ForwardType.SAC_Q,
                            obs=repeated_obs,
                            actions=sampled_actions,
                            shared_feature=None,
                            **dsrl_kwargs,
                        ).reshape(batch_size, num_samples, -1)
                        sampled_q_values = sampled_q_values.transpose(0, 1)
                    else:
                        all_qf_next_target = self.target_model(
                            forward_type=ForwardType.SAC_Q,
                            obs=next_obs,
                            actions=next_state_actions,
                            shared_feature=None,
                            **dsrl_kwargs,
                        )
                        sampled_q_values = [all_qf_next_target]
                        for _ in range(num_samples - 1):
                            sampled_actions, _, _ = bootstrap_actor(
                                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
                            )
                            sampled_q_values.append(
                                self.target_model(
                                    forward_type=ForwardType.SAC_Q,
                                    obs=next_obs,
                                    actions=sampled_actions,
                                    shared_feature=None,
                                    **dsrl_kwargs,
                                )
                            )
                        sampled_q_values = torch.stack(sampled_q_values, dim=0)
                    if reduction == "mean":
                        all_qf_next_target = sampled_q_values.mean(dim=0)
                    else:
                        raise ValueError(f"Unsupported q_vr_reduction: {reduction!r}")
                else:
                    all_qf_next_target = self.target_model(
                        forward_type=ForwardType.SAC_Q,
                        obs=next_obs,
                        actions=next_state_actions,
                        shared_feature=None,
                        **dsrl_kwargs,
                    )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        squared_td_error = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
            reduction="none",
        )
        valid = batch.get("valid")
        if valid is not None:
            # Official sequence replay masks Q loss when the H-step sample has
            # crossed a terminal before its final transition. It intentionally
            # keeps the full-batch denominator rather than renormalizing.
            valid_last = valid[..., -1].to(squared_td_error.dtype).unsqueeze(-1)
            squared_td_error = squared_td_error * valid_last
        critic_loss = squared_td_error.mean()
        critic_metrics = {}
        if collect_metrics:
            if bool(self.cfg.algorithm.get("reduce_metric_cuda_sync", False)):
                q_stats = (
                    torch.stack(
                        (
                            all_data_q_values.mean(),
                            all_data_q_values.min(),
                            all_data_q_values.max(),
                            all_data_q_values.std(unbiased=False),
                            target_q_values.mean(),
                            target_q_values.min(),
                            target_q_values.max(),
                            target_q_values.std(unbiased=False),
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                critic_metrics = {
                    "q_data": q_stats[0],
                    "q_mean": q_stats[0],
                    "q_min": q_stats[1],
                    "q_max": q_stats[2],
                    "q_std": q_stats[3],
                    "target_q_mean": q_stats[4],
                    "target_q_min": q_stats[5],
                    "target_q_max": q_stats[6],
                    "target_q_std": q_stats[7],
                }
            else:
                critic_metrics = {
                    "q_data": all_data_q_values.mean().item(),
                    "q_mean": all_data_q_values.mean().item(),
                    "q_min": all_data_q_values.min().item(),
                    "q_max": all_data_q_values.max().item(),
                    "q_std": all_data_q_values.std(unbiased=False).item(),
                    "target_q_mean": target_q_values.mean().item(),
                    "target_q_min": target_q_values.min().item(),
                    "target_q_max": target_q_values.max().item(),
                    "target_q_std": target_q_values.std(unbiased=False).item(),
                }
        return critic_loss, critic_metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch, collect_metrics: bool = True):
        """Compute OGPO's group-relative Q-guided policy objective."""
        states = batch["curr_obs"]["states"]
        actor_batch_size = min(
            int(self.cfg.algorithm.get("ppo_batch_size", states.shape[0])),
            states.shape[0],
        )
        states = states[:actor_batch_size]
        group_size = int(self.cfg.algorithm.get("group_size", 32))
        repeated_states = (
            states.unsqueeze(0)
            .expand(group_size, *states.shape)
            .reshape(group_size * actor_batch_size, -1)
        )
        grouped_obs = {"states": repeated_states}

        with torch.no_grad():
            actions, chains, old_log_pi = self.target_model(
                forward_type=ForwardType.OGPO_SAMPLE, obs=grouped_obs
            )
            q_values = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=grouped_obs,
                actions=actions.detach(),
            )
            q_values = q_values.reshape(group_size, actor_batch_size, -1)
            advantages, aggregated_q = compute_group_relative_advantages(
                q_values,
                aggregation=str(self.cfg.algorithm.get("actor_agg_q", "mean")),
                normalize=bool(self.cfg.algorithm.get("normalize_group", False)),
                strategy=str(self.cfg.algorithm.get("adv_strategy", "vanilla")),
            )

        log_pi = self.model(
            forward_type=ForwardType.OGPO_LOGPROB,
            obs=grouped_obs,
            chain=chains,
        )
        normalization = 1.0
        if bool(self.cfg.algorithm.get("normalize_action_logprob", True)):
            normalization *= float(
                self.cfg.actor.model.action_dim
                * self.cfg.actor.model.get("num_action_chunks", 1)
            )
        if bool(self.cfg.algorithm.get("normalize_denoising_logprob", True)):
            # The full path contains p(x_0) plus one density per SDE step.
            normalization *= float(self.cfg.actor.model.denoising_steps + 1)
        log_pi = (log_pi / normalization).reshape(group_size, actor_batch_size)
        old_log_pi = (old_log_pi / normalization).reshape(group_size, actor_batch_size)
        log_ratio = log_pi - old_log_pi
        ratio = torch.exp(log_ratio)
        clip_epsilon = float(self.cfg.algorithm.get("clip_epsilon", 0.01))
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        pg_loss = -torch.minimum(
            ratio * advantages.detach(), clipped_ratio * advantages.detach()
        ).mean()

        bc_loss = self._flow_bc_loss(self._online_bc_batch(batch))
        bc_coeff = float(self.cfg.algorithm.get("bc_coeff", 1.0))
        actor_loss = pg_loss + bc_coeff * bc_loss
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        entropy = -log_pi.mean()
        metrics = {}
        if collect_metrics:
            metrics = {
                "pg_loss": float(pg_loss.detach()),
                "bc_loss": float(bc_loss.detach()),
                "q_pi": float(aggregated_q.mean()),
                "adv_mean": float(advantages.mean()),
                "adv_std": float(advantages.std(unbiased=False)),
                "adv_min": float(advantages.min()),
                "adv_max": float(advantages.max()),
                "ratio": float(ratio.mean().detach()),
                "ratio_min": float(ratio.min().detach()),
                "ratio_max": float(ratio.max().detach()),
                "clip_fraction": float(
                    ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon))
                    .float()
                    .mean()
                    .detach()
                ),
                "approx_kl": float(approx_kl.detach()),
                "log_prob": float(log_pi.mean().detach()),
                "log_prob_min": float(log_pi.min().detach()),
                "log_prob_max": float(log_pi.max().detach()),
                "old_log_prob": float(old_log_pi.mean().detach()),
                "action_mean": float(actions.mean()),
                "action_std": float(actions.std(unbiased=False)),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                "q_pi_min": float(aggregated_q.min()),
                "q_pi_max": float(aggregated_q.max()),
                "success_buffer_size": float(len(self._success_bc_samples)),
                "success_buffer_min_samples": float(self._success_buffer_min_samples()),
                "using_success_bc": float(self._using_success_bc),
            }
        return actor_loss, entropy, metrics

    def _cache_ogpo_parameter_partitions(self) -> None:
        """Cache aligned actor/critic parameters for clipping and target EMA."""
        if not getattr(self, "target_model_initialized", False):
            return
        target_parameters = dict(self.target_model.named_parameters())
        partitions = {True: [], False: []}
        trainable_partitions = {True: [], False: []}
        for name, online_parameter in self.model.named_parameters():
            if name not in target_parameters:
                raise RuntimeError(f"Target parameter is missing: {name}")
            is_actor = "q_head" not in name
            partitions[is_actor].append((online_parameter, target_parameters.pop(name)))
            trainable_partitions[is_actor].append(online_parameter)
        if target_parameters:
            raise RuntimeError(
                "Online model is missing target parameters: "
                f"{sorted(target_parameters)[:3]}"
            )
        self._ogpo_parameter_pairs = partitions
        self._ogpo_trainable_parameters = trainable_partitions

    def _clip_ogpo_grad_norm(self, *, actor: bool, max_norm: float) -> torch.Tensor:
        """Clip only the active OGPO partition on the single-GPU fast path."""
        if not hasattr(self, "_ogpo_trainable_parameters"):
            self._cache_ogpo_parameter_partitions()
        if self._world_size == 1:
            return torch.nn.utils.clip_grad_norm_(
                self._ogpo_trainable_parameters[actor], max_norm=max_norm
            )
        return self.model.clip_grad_norm_(max_norm=max_norm)

    def _soft_update_ogpo_target(self, *, actor: bool) -> None:
        """Polyak-update only the official actor or critic target partition."""
        if not hasattr(self, "_ogpo_parameter_pairs"):
            self._cache_ogpo_parameter_partitions()
        tau = float(self.cfg.algorithm.get("actor_tau" if actor else "tau", 0.05))
        parameter_pairs = self._ogpo_parameter_pairs[actor]
        online_parameters = [pair[0] for pair in parameter_pairs]
        target_parameters = [pair[1] for pair in parameter_pairs]
        with torch.no_grad():
            torch._foreach_mul_(target_parameters, 1.0 - tau)
            torch._foreach_add_(target_parameters, online_parameters, alpha=tau)

    def update_one_epoch(
        self, train_actor: bool = True, collect_metrics: bool = True
    ) -> dict[str, Any]:
        """Run one official-order OGPO update: actor EMA, then critic EMA."""
        per_rank_batch_size = self.cfg.actor.global_batch_size // self._world_size
        micro_batch_size = _get_ogpo_online_micro_batch_size(self.cfg, self._world_size)
        global_batch = next(self.buffer_dataloader_iter)
        micro_batches = split_dict_to_chunk(
            global_batch,
            per_rank_batch_size // micro_batch_size,
        )
        for index, batch in enumerate(micro_batches):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            micro_batches[index] = batch

        metrics: dict[str, Any] = {}
        use_cuda_timers = bool(self.cfg.actor.get("ogpo_cuda_timers", False))
        actor_events = None
        critic_events = None
        update_actor = train_actor and self.update_step % self.critic_actor_ratio == 0
        if update_actor:
            if use_cuda_timers:
                actor_events = (torch.cuda.Event(True), torch.cuda.Event(True))
                actor_events[0].record()
            if self._world_size > 1:
                self.qf_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            actor_losses = []
            entropies = []
            actor_metrics: dict[str, list[Any]] = {}
            for batch in micro_batches:
                actor_loss, entropy, values = self.forward_actor(
                    batch, collect_metrics=collect_metrics
                )
                (actor_loss / self.gradient_accumulation).backward()
                if collect_metrics:
                    actor_losses.append(float(actor_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    append_to_dict(actor_metrics, values)
            actor_grad_norm = self._clip_ogpo_grad_norm(
                actor=True, max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self._soft_update_ogpo_target(actor=True)
            if actor_events is not None:
                actor_events[1].record()
            if collect_metrics:
                metrics.update(
                    {
                        "sac/actor_loss": np.mean(actor_losses),
                        "sac/alpha_loss": 0.0,
                        "sac/alpha": self.entropy_temp.alpha,
                        "actor/lr": self.optimizer.param_groups[0]["lr"],
                        "actor/grad_norm": actor_grad_norm,
                        "actor/entropy": np.mean(entropies),
                        **{
                            f"actor/{key}": np.mean(value)
                            for key, value in actor_metrics.items()
                        },
                    }
                )

        if use_cuda_timers:
            critic_events = (torch.cuda.Event(True), torch.cuda.Event(True))
            critic_events[0].record()
        self.optimizer.zero_grad(set_to_none=True)
        self.qf_optimizer.zero_grad(set_to_none=True)
        critic_losses = []
        critic_metrics: dict[str, list[Any]] = {}
        for batch in micro_batches:
            critic_loss, values = self.forward_critic(
                batch, collect_metrics=collect_metrics
            )
            (critic_loss / self.gradient_accumulation).backward()
            if collect_metrics:
                critic_losses.append(float(critic_loss.detach()))
                append_to_dict(critic_metrics, values)
        critic_grad_norm = self._clip_ogpo_grad_norm(
            actor=False, max_norm=self.cfg.actor.critic_optim.clip_grad
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()
        self._soft_update_ogpo_target(actor=False)
        success_q_loss = None
        if bool(self.cfg.algorithm.get("use_success_buffer_q", False)):
            success_batch = self._online_success_q_batch(per_rank_batch_size)
            if success_batch is not None:
                self.qf_optimizer.zero_grad(set_to_none=True)
                success_q_loss, _ = self.forward_critic(
                    success_batch, collect_metrics=False
                )
                success_q_loss.backward()
                self._clip_ogpo_grad_norm(
                    actor=False, max_norm=self.cfg.actor.critic_optim.clip_grad
                )
                self.qf_optimizer.step()
                self.qf_lr_scheduler.step()
                self._soft_update_ogpo_target(actor=False)
        if collect_metrics:
            metrics.update(
                {
                    "sac/critic_loss": np.mean(critic_losses),
                    "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
                    "critic/grad_norm": critic_grad_norm,
                    **{
                        f"critic/{key}": np.mean(value)
                        for key, value in critic_metrics.items()
                    },
                }
            )
            if success_q_loss is not None:
                metrics["critic/success_buffer_loss"] = float(success_q_loss.detach())
        if critic_events is not None:
            critic_events[1].record()
            torch.cuda.synchronize(self.device)
            metrics["profile/critic_ms_per_update"] = critic_events[0].elapsed_time(
                critic_events[1]
            )
            if actor_events is not None:
                metrics["profile/actor_training_ms_per_update"] = actor_events[
                    0
                ].elapsed_time(actor_events[1])
        return metrics

    def _run_online_training(self) -> dict[str, Any]:
        """Run OGPO online updates without changing the shared SAC loop."""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, "
                "skipping training"
            )
            return {}

        train_actor_steps = max(
            min_buffer_size, self.cfg.algorithm.get("train_actor_steps", 0)
        )
        train_actor = self.replay_buffer.is_ready(train_actor_steps)
        micro_batch_size = _get_ogpo_online_micro_batch_size(self.cfg, self._world_size)
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size // micro_batch_size // self._world_size
        )

        self.model.train()
        metrics: dict[str, list[Any]] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        actor_epochs = [
            epoch
            for epoch in range(update_epoch)
            if train_actor and (self.update_step + epoch) % self.critic_actor_ratio == 0
        ]
        last_actor_epoch = actor_epochs[-1] if actor_epochs else None
        for epoch in range(update_epoch):
            collect_metrics = epoch == update_epoch - 1 or epoch == last_actor_epoch
            update_metrics = self.update_one_epoch(
                train_actor=train_actor, collect_metrics=collect_metrics
            )
            if update_metrics:
                append_to_dict(metrics, update_metrics)
            self.update_step += 1
        return self.process_train_metrics(metrics)

    @Worker.timer("run_training")
    def run_training(self):
        """Dispatch to OGPO BC or online Q/PPO training by runner step."""
        bc_steps = int(self.cfg.algorithm.get("bc_runner_steps", 0))
        current_step = int(getattr(self, "version", 0))
        if current_step < bc_steps and not self._force_bc_finished:
            return self._run_bc_training()

        self._enter_online_phase()
        envs_per_step = (
            int(self.cfg.env.train.total_num_envs)
            * int(self.cfg.env.train.rollout_epoch)
            * int(self.cfg.env.train.max_steps_per_rollout_epoch)
        )
        self._online_env_steps += envs_per_step
        start_training = int(self.cfg.algorithm.get("start_training_env_steps", 10_000))
        profile_enabled = bool(self.cfg.actor.get("ogpo_torch_profiler", False))
        profile_start = int(
            self.cfg.actor.get("ogpo_profiler_start_env_steps", start_training)
        )
        profile_limit = int(self.cfg.actor.get("ogpo_profiler_trace_count", 1))
        should_profile = (
            profile_enabled
            and self._online_env_steps >= profile_start
            and self._ogpo_profile_traces < profile_limit
        )
        profile_context = nullcontext()
        if should_profile:
            profile_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        training_events = None
        if bool(self.cfg.actor.get("ogpo_cuda_timers", False)):
            training_events = (torch.cuda.Event(True), torch.cuda.Event(True))
            training_events[0].record()
        if self._online_env_steps < start_training:
            metrics = {}
        else:
            with profile_context as profiler:
                metrics = self._run_online_training()
            if should_profile:
                profile_dir = Path(os.environ.get("OGPO_PROFILE_DIR", "ogpo_profiles"))
                profile_dir.mkdir(parents=True, exist_ok=True)
                trace_path = profile_dir / (
                    f"actor_rank{self._rank}_env{self._online_env_steps}.json"
                )
                profiler.export_chrome_trace(str(trace_path))
                self._ogpo_profile_traces += 1
                self.log_info(f"Exported OGPO profiler trace to {trace_path}")
        if training_events is not None:
            training_events[1].record()
            torch.cuda.synchronize(self.device)
            metrics["profile/full_training_cuda_ms"] = training_events[0].elapsed_time(
                training_events[1]
            )
            metrics["profile/gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
            metrics["profile/gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(self.device) / 1024**2
            )
        metrics["ogpo/phase"] = 1.0
        metrics["ogpo/bc_update_steps"] = float(self._bc_update_steps)
        metrics["ogpo/online_env_steps"] = float(self._online_env_steps)
        return metrics

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        """Save SAC state plus sliding windows and successful-episode reservoir."""
        super().save_checkpoint(save_base_path, step)
        state_path = os.path.join(
            save_base_path, f"ogpo_online_state_rank_{self._rank}.pt"
        )
        torch.save(
            {
                "primitive_windows": self._primitive_windows,
                "pending_replay_sequences": self._pending_replay_sequences,
                "episode_ids": self._episode_ids,
                "episode_success": self._episode_success,
                "episode_success_labels": self._episode_success_labels,
                "success_bc_samples": self._success_bc_samples,
            },
            state_path,
        )

    def load_checkpoint(self, load_base_path: str) -> None:
        """Restore model, replay, and OGPO online sequence state."""
        super().load_checkpoint(load_base_path)
        state_path = os.path.join(
            load_base_path, f"ogpo_online_state_rank_{self._rank}.pt"
        )
        if not os.path.exists(state_path):
            self._clear_online_sequence_state()
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self._primitive_windows = state["primitive_windows"]
        self._pending_replay_sequences = state["pending_replay_sequences"]
        self._episode_ids = state["episode_ids"]
        self._episode_success = state["episode_success"]
        self._episode_success_labels = state["episode_success_labels"]
        self._success_bc_samples = state["success_bc_samples"]
