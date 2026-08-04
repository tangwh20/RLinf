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

"""Unit tests for the state-based embodied OGPO primitives."""

import copy

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.data.ogpo_io_struct import (
    OGPOEmbodiedRolloutResult,
    OGPOTrajectory,
    _extract_chunk_successes,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.flow_policy.flow_policy import (
    FlowStateConfig,
    FlowStatePolicy,
)
from rlinf.models.embodiment.modules.ogpo import OGPOMultiQHead
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.ogpo_embodied_runner import (
    OGPOEmbodiedRunner,
    compute_ogpo_evaluate_metrics,
)
from rlinf.scheduler.cluster.utils import extract_dataclass_tensor_fields
from rlinf.workers.actor.fsdp_ogpo_policy_worker import (
    EmbodiedOGPOFSDPPolicy,
    _ActionChunkDataset,
    _get_ogpo_online_micro_batch_size,
    _OGPOSingleGPUStrategyAdapter,
    _use_ogpo_single_gpu_fast_path,
    compute_discounted_chunk_return,
    compute_group_relative_advantages,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def test_conservative_group_relative_advantages() -> None:
    """Conservative advantages keep only Q-ensemble sign consensus."""
    q_values = torch.tensor(
        [
            [[3.0, 4.0], [1.0, 4.0]],
            [[1.0, 2.0], [3.0, 2.0]],
        ]
    )
    advantages, aggregated = compute_group_relative_advantages(
        q_values, strategy="conservative"
    )

    torch.testing.assert_close(aggregated, q_values.mean(dim=-1))
    torch.testing.assert_close(advantages, torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))


def test_conservative_advantages_ignore_vanilla_group_normalization() -> None:
    """Official conservative advantages are not group-normalized."""
    q_values = torch.tensor([[[4.0, 8.0]], [[2.0, 2.0]], [[0.0, 0.0]]])

    advantages, _ = compute_group_relative_advantages(
        q_values, normalize=True, strategy="conservative"
    )

    torch.testing.assert_close(advantages, torch.tensor([[2.0], [0.0], [-2.0]]))


def test_primitive_transition_collection_accepts_numpy_actions() -> None:
    """D4RL action chunks cross the worker boundary as NumPy arrays."""
    result = OGPOEmbodiedRolloutResult()
    tensor = torch.zeros(1, 4, 1)
    result.append_primitive_transitions(
        curr_states=tensor,
        next_states=tensor,
        actions=np.zeros((1, 4, 2), dtype=np.float32),
        rewards=tensor.squeeze(-1),
        successes=tensor.squeeze(-1).bool(),
        terminations=tensor.squeeze(-1).bool(),
        truncations=tensor.squeeze(-1).bool(),
    )

    assert isinstance(result.primitive_actions[0], torch.Tensor)
    assert result.primitive_actions[0].shape == (1, 4, 2)


def test_ogpo_split_preserves_declared_trajectory_fields() -> None:
    """Collective serialization only transfers declared dataclass tensor fields."""
    result = OGPOEmbodiedRolloutResult(source_rank=3)
    tensor = torch.zeros(2, 4, 1)
    result.append_primitive_transitions(
        curr_states=tensor,
        next_states=tensor,
        actions=torch.zeros(2, 4, 2),
        rewards=tensor.squeeze(-1),
        successes=tensor.squeeze(-1).bool(),
        terminations=tensor.squeeze(-1).bool(),
        truncations=tensor.squeeze(-1).bool(),
    )

    trajectories = result.to_splited_trajectories(2)

    assert all(isinstance(item, OGPOTrajectory) for item in trajectories)
    assert all(item.source_rank == 3 for item in trajectories)
    assert all(
        item.primitive_curr_states.shape == (1, 1, 4, 1) for item in trajectories
    )
    _, _, metadata = extract_dataclass_tensor_fields(trajectories[0])
    serialized_fields = {field_name for field_name, _, _ in metadata}
    assert set(OGPOEmbodiedRolloutResult._PRIMITIVE_FIELDS) <= serialized_fields


def test_auto_reset_terminal_success_reaches_ogpo_success_buffer() -> None:
    """Auto-reset final_info preserves the terminal OGPO success label."""
    infos_list = [
        {"success": torch.zeros(1)},
        {"success": torch.zeros(1)},
        {"success": torch.zeros(1)},
        {
            "final_info": {"success": torch.ones(1)},
            "_final_info": torch.ones(1, dtype=torch.bool),
        },
    ]

    successes = _extract_chunk_successes(infos_list, batch_size=1)
    torch.testing.assert_close(successes, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    worker = _sequence_worker()
    trajectory = _primitive_trajectory(start=0, success=False)
    trajectory.primitive_successes = successes.bool().unsqueeze(0)
    completed = worker._build_sliding_sequence_trajectory(trajectory)

    assert completed is not None
    assert len(worker._success_bc_samples) == 1


def test_flow_bc_and_target_chain_logprob_have_gradients() -> None:
    """BC and target-chain PPO primitives support finite actor gradients."""
    cfg = FlowStateConfig(
        obs_dim=5,
        action_dim=3,
        add_q_head=True,
        num_q_heads=2,
        denoising_steps=2,
        d_model=16,
        n_head=4,
        n_layers=1,
        noise_std_train=0.05,
        flow_actor_type="OGPOFlowMLPActor",
        actor_hidden_dims=(16, 16, 16, 16),
        critic_hidden_dims=(16, 16, 16, 16),
        time_embedding_dim=8,
    )
    model = FlowStatePolicy(cfg)
    target = copy.deepcopy(model).requires_grad_(False)
    observations = {"states": torch.randn(4, cfg.obs_dim)}
    dataset_actions = torch.rand(4, cfg.action_dim) * 2.0 - 1.0

    predicted, velocity_target = model(
        forward_type=ForwardType.OGPO_BC,
        obs=observations,
        actions=dataset_actions,
    )
    bc_loss = (predicted - velocity_target).square().mean()
    bc_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, chain, old_log_prob = target(
            forward_type=ForwardType.OGPO_SAMPLE, obs=observations
        )
    log_prob = model(
        forward_type=ForwardType.OGPO_LOGPROB,
        obs=observations,
        chain=chain,
    )
    policy_loss = -((log_prob - old_log_prob) * torch.arange(4.0)).mean()
    policy_loss.backward()

    assert chain.shape == (4, cfg.denoising_steps + 1, cfg.action_dim)
    assert torch.isfinite(log_prob).all()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_ode_and_sde_bon_sampling_interfaces() -> None:
    """ODE and Q-filtered SDE evaluation both preserve policy output shapes."""
    cfg = FlowStateConfig(
        obs_dim=5,
        action_dim=3,
        add_q_head=True,
        num_q_heads=4,
        denoising_steps=2,
        d_model=16,
        n_head=4,
        n_layers=1,
        best_of_n=3,
        subsample_bon=True,
        flow_actor_type="OGPOFlowMLPActor",
        actor_hidden_dims=(16, 16, 16, 16),
        critic_hidden_dims=(16, 16, 16, 16),
        time_embedding_dim=8,
    )
    model = FlowStatePolicy(cfg)
    observations = {"states": torch.randn(4, cfg.obs_dim)}

    ode_actions, ode_result = model.predict_action_batch(
        observations, sampling_mode="ode"
    )
    sde_actions, sde_result = model.predict_action_batch(
        observations, sampling_mode="sde"
    )

    assert ode_actions.shape == sde_actions.shape == (4, 1, cfg.action_dim)
    assert torch.isfinite(ode_actions).all()
    assert torch.isfinite(sde_actions).all()
    assert ode_result["prev_logprobs"].shape == (4, 1)
    assert sde_result["prev_logprobs"].shape == (4, 1)


def test_ogpo_eval_metrics_keep_uncertainty_and_success_length() -> None:
    """Aggregation matches OGPO means and retains std/SEM diagnostics."""
    metrics = compute_ogpo_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                "return": torch.tensor([10.0, -20.0, 30.0, -40.0]),
                "normalized_return": torch.tensor([60.0, 10.0, 80.0, 0.0]),
                "length": torch.tensor([100.0, 200.0, 120.0, 200.0]),
            }
        ]
    )

    assert metrics["num_trajectories"] == 4
    assert metrics["success"] == 0.5
    assert metrics["success_count"] == 2
    assert metrics["length"] == 110.0
    assert metrics["length_all"] == 155.0
    assert metrics["return_std"] > 0.0
    assert metrics["normalized_return_sem"] > 0.0


def test_ogpo_early_bc_stop_preserves_online_transition_budget() -> None:
    """A clipped BC phase shortens total runtime, not the 500k online budget."""

    class _Waitable:
        def wait(self):
            return None

    class _Actor:
        finished = False

        def finish_ogpo_bc(self):
            self.finished = True
            return _Waitable()

    class _Logger:
        def info(self, *args, **kwargs):
            return None

    runner = object.__new__(OGPOEmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "algorithm": {
                "bc_runner_steps": 500,
                "bc_updates_per_step": 100,
                "ogpo_online_env_steps": 500_000,
                "ogpo_eval_interval_bc_updates": 1_000,
                "ogpo_eval_interval_env_steps": 20_000,
                "ogpo_log_interval_bc_updates": 5_000,
                "ogpo_log_interval_env_steps": 5_000,
            },
            "env": {
                "train": {
                    "total_num_envs": 32,
                    "rollout_epoch": 1,
                    "max_steps_per_rollout_epoch": 4,
                }
            },
        }
    )
    runner.actor = _Actor()
    runner.logger = _Logger()
    runner.global_step = 120
    runner.max_steps = 16_125
    runner._bc_finished = False
    runner._actual_bc_runner_steps = None

    runner._finish_bc_phase()

    assert runner.actor.finished
    assert runner._progress() == (12_000, 0, 12_000)
    assert runner.max_steps == 120 + 3_907

    runner.global_step += 157
    assert runner._progress() == (12_000, 20_096, 32_096)
    assert runner._metric_step(276) == 32_096
    assert runner._should_evaluate()
    assert runner._should_log_step(0)


def test_default_embodied_metric_step_is_unchanged() -> None:
    """Non-OGPO runners retain the native RLinf runner-step x-axis."""
    runner = object.__new__(EmbodiedRunner)
    assert runner._metric_step(123) == 123


def test_ogpo_single_gpu_fast_path_is_strictly_gated() -> None:
    """The plain-module path is opt-in and falls back for other configurations."""
    cfg = OmegaConf.create(
        {
            "actor": {
                "ogpo_single_gpu_fast_path": True,
                "enable_offload": False,
                "fsdp_config": {
                    "strategy": "fsdp",
                    "sharding_strategy": "no_shard",
                    "cpu_offload": False,
                },
            }
        }
    )

    assert _use_ogpo_single_gpu_fast_path(cfg, world_size=1)
    cfg.actor.ogpo_single_gpu_fast_path = False
    assert not _use_ogpo_single_gpu_fast_path(cfg, world_size=2)
    cfg.actor.ogpo_single_gpu_fast_path = True
    assert not _use_ogpo_single_gpu_fast_path(cfg, world_size=2)
    cfg.actor.fsdp_config.sharding_strategy = "full_shard"
    assert not _use_ogpo_single_gpu_fast_path(cfg, world_size=1)
    cfg.actor.fsdp_config.sharding_strategy = "no_shard"
    cfg.actor.enable_offload = True
    assert not _use_ogpo_single_gpu_fast_path(cfg, world_size=1)


def test_ogpo_single_gpu_strategy_adapter_only_replaces_wrapping() -> None:
    """All non-wrapping operations remain delegated to the native strategy."""

    class _Strategy:
        marker = "native"

    strategy = _Strategy()
    adapter = _OGPOSingleGPUStrategyAdapter(
        strategy, device=torch.device("cpu"), dtype=torch.float32
    )
    model = nn.Linear(2, 1).double()

    wrapped = adapter.wrap_model(model, device_mesh=object())

    assert wrapped is model
    assert wrapped.weight.dtype == torch.float32
    assert adapter.marker == "native"


def test_ogpo_single_gpu_fast_path_uses_full_online_batch() -> None:
    """The local fast path avoids accumulation without changing fallback paths."""
    cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 256,
                "micro_batch_size": 128,
                "ogpo_single_gpu_micro_batch_size": 256,
                "ogpo_single_gpu_fast_path": True,
                "enable_offload": False,
                "fsdp_config": {
                    "strategy": "fsdp",
                    "sharding_strategy": "no_shard",
                    "cpu_offload": False,
                },
            }
        }
    )

    assert _get_ogpo_online_micro_batch_size(cfg, world_size=1) == 256
    assert _get_ogpo_online_micro_batch_size(cfg, world_size=2) == 128
    cfg.actor.ogpo_single_gpu_fast_path = False
    assert _get_ogpo_online_micro_batch_size(cfg, world_size=1) == 128


def test_ogpo_transition_checkpoint_restores_clean_online_phase() -> None:
    """A BC-transition resume clears warmup replay and restores online LR."""

    class _ReplayBuffer:
        cleared = False

        def clear(self):
            self.cleared = True

    class _Scheduler:
        base_lrs = [3.0e-4]

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker.cfg = OmegaConf.create({"algorithm": {"online_actor_lr": 4.5e-5}})
    worker.optimizer = type("Optimizer", (), {"param_groups": [{"lr": 3.0e-4}]})()
    worker.qf_optimizer = type("Optimizer", (), {"param_groups": [{"lr": 3.0e-4}]})()
    worker.lr_scheduler = _Scheduler()
    worker.replay_buffer = _ReplayBuffer()
    worker.buffer_dataloader = [object()]
    worker._online_phase_initialized = False
    worker._primitive_windows = {}
    worker._pending_replay_sequences = {}
    worker._episode_ids = {}
    worker._episode_success = {}
    worker._episode_success_labels = {}
    from collections import deque

    worker._success_bc_samples = deque(maxlen=100)
    worker._using_success_bc = False

    def reset_optimizers():
        for optimizer in (worker.optimizer, worker.qf_optimizer):
            optimizer.param_groups[0]["lr"] = 4.5e-5
        worker.lr_scheduler.base_lrs = [4.5e-5]

    worker._reset_online_optimizers = reset_optimizers
    worker.restore_ogpo_training_state(
        bc_update_steps=12_000, online_env_steps=0, bc_finished=True
    )

    assert worker._bc_update_steps == 12_000
    assert worker._online_env_steps == 0
    assert worker._force_bc_finished
    assert worker._online_phase_initialized
    assert worker.replay_buffer.cleared
    assert worker.optimizer.param_groups[0]["lr"] == 4.5e-5
    assert worker.qf_optimizer.param_groups[0]["lr"] == 4.5e-5
    assert worker.lr_scheduler.base_lrs == [4.5e-5]


def test_finish_bc_enters_online_before_next_rollout() -> None:
    """BC rollouts are cleared before, not after, the first online rollout."""
    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker._force_bc_finished = False
    transitions = []
    worker._enter_online_phase = lambda: transitions.append("entered")

    worker.finish_ogpo_bc()

    assert worker._force_bc_finished
    assert transitions == ["entered"]


def test_bc_updates_target_actor_with_official_tau(monkeypatch) -> None:
    """Every BC optimizer step is followed by the official actor EMA update."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def clip_grad_norm_(self, max_norm):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"bc_updates_per_step": 3, "actor_tau": 0.05},
            "actor": {"optim": {"clip_grad": 1000.0}},
        }
    )
    worker.model = _Model()
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1.0
    )
    worker._next_offline_batch = lambda: {}
    worker._flow_bc_loss = lambda _: worker.model.weight.square()
    worker._clip_ogpo_grad_norm = lambda **kwargs: torch.nn.utils.clip_grad_norm_(
        worker.model.parameters(), kwargs["max_norm"]
    )
    worker._bc_update_steps = 0
    worker._online_env_steps = 0
    target_updates = []
    worker._soft_update_ogpo_target = lambda *, actor: target_updates.append(actor)
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_ogpo_policy_worker.all_reduce_dict",
        lambda metrics, op: metrics,
    )

    metrics = worker._run_bc_training()

    assert target_updates == [True, True, True]
    assert metrics["ogpo/bc_update_steps"] == 3.0


def test_cached_ogpo_partitions_preserve_ema_and_clipping_semantics() -> None:
    """Foreach EMA and scoped clipping touch only the selected partition."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(1, 1, bias=False)
            self.q_head = nn.Linear(1, 1, bias=False)

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker.cfg = OmegaConf.create({"algorithm": {"actor_tau": 0.25, "tau": 0.5}})
    worker.model = _Model()
    worker.target_model = _Model()
    worker.target_model_initialized = True
    worker._world_size = 1
    with torch.no_grad():
        worker.model.actor.weight.fill_(2.0)
        worker.model.q_head.weight.fill_(4.0)
        worker.target_model.actor.weight.zero_()
        worker.target_model.q_head.weight.zero_()

    worker._cache_ogpo_parameter_partitions()
    worker._soft_update_ogpo_target(actor=True)
    torch.testing.assert_close(
        worker.target_model.actor.weight,
        torch.full_like(worker.model.actor.weight, 0.5),
    )
    torch.testing.assert_close(
        worker.target_model.q_head.weight,
        torch.zeros_like(worker.model.q_head.weight),
    )

    worker._soft_update_ogpo_target(actor=False)
    torch.testing.assert_close(
        worker.target_model.q_head.weight,
        torch.full_like(worker.model.q_head.weight, 2.0),
    )
    worker.model.actor.weight.grad = torch.full_like(worker.model.actor.weight, 3.0)
    worker.model.q_head.weight.grad = torch.full_like(worker.model.q_head.weight, 100.0)
    worker._clip_ogpo_grad_norm(actor=True, max_norm=1.0)
    assert worker.model.actor.weight.grad.norm() <= 1.0
    torch.testing.assert_close(
        worker.model.q_head.weight.grad,
        torch.full_like(worker.model.q_head.weight, 100.0),
    )


def test_ogpo_checkpoint_interval_detects_crossed_boundary() -> None:
    """Checkpoint cadence is robust when vector steps skip an exact boundary."""
    runner = object.__new__(OGPOEmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "algorithm": {
                "bc_runner_steps": 500,
                "bc_updates_per_step": 100,
                "ogpo_online_env_steps": 500_000,
                "ogpo_save_interval_env_steps": 100_000,
            },
            "env": {
                "train": {
                    "total_num_envs": 48,
                    "rollout_epoch": 1,
                    "max_steps_per_rollout_epoch": 4,
                }
            },
        }
    )
    runner._bc_finished = True
    runner._actual_bc_runner_steps = 100
    runner.global_step = 100 + 521  # 100,032 online transitions.
    runner._online_target = lambda: 500_000
    runner._should_evaluate = lambda: False
    saved = []
    runner._save_checkpoint = lambda: saved.append(True)

    runner._maybe_eval_and_checkpoint(0)

    assert saved == [True]


def test_ogpo_four_step_action_chunk_shapes() -> None:
    """The flow actor and Q ensemble consume a flattened four-action chunk."""
    cfg = FlowStateConfig(
        obs_dim=5,
        action_dim=3,
        num_action_chunks=4,
        add_q_head=True,
        num_q_heads=2,
        denoising_steps=2,
        d_model=16,
        n_head=4,
        n_layers=1,
        use_tapered_noise=True,
        error_correct_sde_to_ode=True,
        noise_std_train=0.05,
        noise_std_rollout=0.05,
        flow_actor_type="OGPOFlowMLPActor",
        actor_hidden_dims=(16, 16, 16, 16),
        critic_hidden_dims=(16, 16, 16, 16),
        time_embedding_dim=8,
    )
    model = FlowStatePolicy(cfg)
    observations = {"states": torch.randn(4, cfg.obs_dim)}
    actions, result = model.predict_action_batch(observations, sampling_mode="sde")

    assert actions.shape == (4, 4, 3)
    assert result["forward_inputs"]["action"].shape == (4, 12)
    q_values = model(
        forward_type=ForwardType.SAC_Q,
        obs=observations,
        actions=result["forward_inputs"]["action"],
    )
    assert q_values.shape == (4, 2)


def test_discounted_four_step_td_components() -> None:
    """OGPO uses a discounted four-step reward and gamma**4 bootstrap."""
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    gamma = 0.5
    result = compute_discounted_chunk_return(rewards, gamma, horizon=4)
    torch.testing.assert_close(result, torch.tensor([[3.25]]))
    assert gamma**4 == 0.0625


def test_offline_dataset_builds_arbitrary_start_action_chunks() -> None:
    """Official BC sampling keeps all starts and masks actions after terminals."""

    class _Dataset:
        observations = torch.arange(30.0).reshape(6, 5).numpy()
        actions = torch.arange(18.0).reshape(6, 3).numpy()
        dones_float = torch.tensor([0, 0, 0, 1, 0, 1]).numpy()

        def __len__(self):
            return len(self.observations)

    chunked = _ActionChunkDataset(_Dataset(), horizon=4)
    assert len(chunked) == 3
    sample = chunked[1]
    assert sample["actions"].shape == (12,)
    torch.testing.assert_close(sample["actions"], torch.arange(3.0, 15.0))
    torch.testing.assert_close(sample["valid"], torch.tensor([1.0, 1.0, 1.0, 0.0]))


def _primitive_trajectory(
    *, start: int, success: bool, source_rank: int = 0
) -> OGPOTrajectory:
    values = torch.arange(start, start + 4, dtype=torch.float32).reshape(1, 1, 4)
    states = values.unsqueeze(-1)
    return OGPOTrajectory(
        source_rank=source_rank,
        primitive_curr_states=states,
        primitive_next_states=states + 1.0,
        primitive_actions=states.repeat(1, 1, 1, 2),
        primitive_rewards=values,
        primitive_successes=torch.tensor(
            [[[False, False, False, success]]], dtype=torch.bool
        ),
        primitive_terminations=torch.zeros(1, 1, 4, dtype=torch.bool),
        primitive_truncations=torch.tensor(
            [[[False, False, False, True]]], dtype=torch.bool
        ),
    )


def _sequence_worker() -> EmbodiedOGPOFSDPPolicy:
    from collections import deque

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker.cfg = OmegaConf.create({"actor": {"model": {"num_action_chunks": 4}}})
    worker._primitive_windows = {}
    worker._pending_replay_sequences = {}
    worker._episode_ids = {}
    worker._episode_success = {}
    worker._episode_success_labels = {}
    worker._success_bc_samples = deque(maxlen=100)
    return worker


def test_completed_episode_replay_keeps_cross_episode_sequences_and_valid() -> None:
    """Replay matches official flat-buffer starts while delaying incomplete episodes."""
    worker = _sequence_worker()

    first = worker._build_sliding_sequence_trajectory(
        _primitive_trajectory(start=0, success=True)
    )
    second = worker._build_sliding_sequence_trajectory(
        _primitive_trajectory(start=4, success=False)
    )

    assert first is not None and first.actions.shape[0] == 1
    assert second is not None and second.actions.shape[0] == 4
    torch.testing.assert_close(
        second.valid[:, 0],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )
    # Four starts belong to the successful first episode: its in-episode
    # sequence plus the three starts that cross into episode two.
    assert len(worker._success_bc_samples) == 4
    assert worker._pending_replay_sequences[(0, 0)] == []


def test_success_bc_buffer_uses_terminal_frame_success() -> None:
    """Transient success does not label an unsuccessful terminal episode."""
    worker = _sequence_worker()
    trajectory = _primitive_trajectory(start=0, success=False)
    trajectory.primitive_successes[0, 0, 1] = True

    completed = worker._build_sliding_sequence_trajectory(trajectory)

    assert completed is not None
    assert len(worker._success_bc_samples) == 0
    assert worker._episode_success_labels[(0, 0, 0)] is False


def test_official_ogpo_network_topology() -> None:
    """The selected state policy uses raw-state 4xMLP actor and 10 independent Qs."""
    cfg = FlowStateConfig(
        obs_dim=5,
        action_dim=3,
        num_action_chunks=4,
        add_q_head=True,
        num_q_heads=10,
        denoising_steps=3,
        flow_actor_type="OGPOFlowMLPActor",
        actor_hidden_dims=(32, 32, 32, 32),
        critic_hidden_dims=(32, 32, 32, 32),
        time_embedding_dim=8,
    )
    model = FlowStatePolicy(cfg)

    assert isinstance(model.backbone, nn.Identity)
    actor_linears = [
        module
        for module in model.flow_actor.velocity_net
        if isinstance(module, nn.Linear)
    ]
    assert len(actor_linears) == 5
    assert model.q_head.num_q_heads == 10
    assert len(model.q_head.weights) == 5
    assert len(model.q_head.norm_weights) == 4
    assert all(weight.shape[0] == 10 for weight in model.q_head.weights)


def test_vectorized_ogpo_q_matches_independent_head_path() -> None:
    """Small-batch vectorization preserves independent-head Q calculations."""
    model = OGPOMultiQHead(
        obs_dim=5,
        action_dim=12,
        hidden_dims=(32, 32, 32, 32),
        num_q_heads=4,
        vectorized_batch_limit=16,
    )
    reference = copy.deepcopy(model)
    reference.vectorized_batch_limit = 0
    states = torch.randn(8, 5)
    actions = torch.randn(8, 12)

    actual = model(states, actions)
    expected = reference(states, actions)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    actual.square().mean().backward()
    expected.square().mean().backward()
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=3e-3, atol=3e-4
        )


def test_vectorized_ogpo_q_loads_legacy_module_list_checkpoint() -> None:
    """Existing ``qs.<head>`` checkpoints migrate to stacked parameters."""
    model = OGPOMultiQHead(
        obs_dim=5,
        action_dim=12,
        hidden_dims=(32, 32, 32, 32),
        num_q_heads=4,
    )
    legacy_state = {}
    for layer_index, module_index in enumerate(model._LINEAR_MODULE_INDICES):
        for head_index in range(model.num_q_heads):
            legacy_state[f"qs.{head_index}.{module_index}.weight"] = model.weights[
                layer_index
            ][head_index].clone()
            legacy_state[f"qs.{head_index}.{module_index}.bias"] = model.biases[
                layer_index
            ][head_index].clone()
    for layer_index, module_index in enumerate(model._NORM_MODULE_INDICES):
        for head_index in range(model.num_q_heads):
            legacy_state[f"qs.{head_index}.{module_index}.weight"] = model.norm_weights[
                layer_index
            ][head_index].clone()
            legacy_state[f"qs.{head_index}.{module_index}.bias"] = model.norm_biases[
                layer_index
            ][head_index].clone()

    restored = copy.deepcopy(model)
    for parameter in restored.parameters():
        parameter.data.zero_()
    restored.load_state_dict(legacy_state, strict=True)

    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)


def test_rollout_state_dict_switches_between_current_and_target() -> None:
    """Online SDE exports EMA weights while ODE evaluation exports current weights."""

    class _Strategy:
        def get_model_state_dict(self, model, cpu_offload=False, full_state_dict=False):
            assert not cpu_offload
            assert not full_state_dict
            return {"source": model}

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker._strategy = _Strategy()
    worker.model = "current"
    worker.target_model = "target"

    worker.set_ogpo_rollout_weight_source("target")
    assert worker.get_rollout_state_dict() == {"source": "target"}
    worker.set_ogpo_rollout_weight_source("current")
    assert worker.get_rollout_state_dict() == {"source": "current"}


def test_online_bc_switches_to_success_reservoir_when_ready() -> None:
    """Online actor BC samples successful episodes after readiness threshold."""
    worker = _sequence_worker()
    worker._world_size = 1
    worker.device = torch.device("cpu")
    worker.cfg = OmegaConf.create(
        {
            "actor": {"global_batch_size": 1, "model": {"num_action_chunks": 1}},
            "algorithm": {"success_buffer_batch_size": 1, "success_buffer_utd": 1},
        }
    )
    sample = {
        "observations": torch.tensor([1.0, 2.0]),
        "actions": torch.tensor([3.0]),
        "valid": torch.tensor([1.0]),
    }
    worker._success_bc_samples.extend([sample, sample])
    replay_batch = {
        "curr_obs": {"states": torch.zeros(1, 2)},
        "actions": torch.zeros(1, 1),
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
    }

    batch = worker._online_bc_batch(replay_batch)

    assert worker._using_success_bc
    torch.testing.assert_close(
        batch["observations"], sample["observations"].unsqueeze(0)
    )
    torch.testing.assert_close(batch["actions"], sample["actions"].unsqueeze(0))


def test_ogpo_online_sequence_state_checkpoint_roundtrip(tmp_path, monkeypatch) -> None:
    """The OGPO replay window and successful BC reservoir survive a resume."""
    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "load_checkpoint", lambda *args: None)
    worker = _sequence_worker()
    worker._rank = 0
    worker._using_success_bc = True
    worker._episode_ids[(3, 2)] = 7
    worker._success_bc_samples.append(
        {
            "observations": torch.tensor([1.0]),
            "actions": torch.tensor([2.0]),
            "valid": torch.tensor([1.0]),
        }
    )
    worker.save_checkpoint(str(tmp_path), step=1)

    restored = _sequence_worker()
    restored._rank = 0
    restored.load_checkpoint(str(tmp_path))

    assert restored._episode_ids == {(3, 2): 7}
    assert len(restored._success_bc_samples) == 1
    torch.testing.assert_close(
        restored._success_bc_samples[0]["actions"], torch.tensor([2.0])
    )


def test_non_ogpo_flow_state_keeps_existing_actor_contract() -> None:
    """OGPO sampling does not alter either existing state-flow actor path."""
    for actor_type in ("FlowTActor", "JaxFlowTActor"):
        cfg = FlowStateConfig(
            obs_dim=5,
            action_dim=3,
            num_action_chunks=1,
            add_q_head=True,
            num_q_heads=2,
            denoising_steps=2,
            d_model=16,
            n_head=4,
            n_layers=1,
            flow_actor_type=actor_type,
        )
        model = FlowStatePolicy(cfg)
        observations = {"states": torch.randn(2, cfg.obs_dim)}

        actions, rollout = model.predict_action_batch(observations)
        assert actions.shape == (2, 1, cfg.action_dim)
        assert rollout["forward_inputs"]["action"].shape == (2, cfg.action_dim)

        sac_actions, log_prob, _ = model(forward_type=ForwardType.SAC, obs=observations)
        assert sac_actions.shape == (2, cfg.action_dim)
        assert log_prob.shape[0] == 2
        assert not model.use_official_ogpo_arch
        assert not hasattr(model.flow_actor, "sample_chain")
