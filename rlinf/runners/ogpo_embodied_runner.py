# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OGPO runner with official Adroit evaluation cadence and metric semantics."""

import json
import math
import os
import time

import torch

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle


def compute_ogpo_evaluate_metrics(
    eval_metrics_list: list[dict], exact_episode_count: int | None = None
) -> dict:
    """Aggregate episode tensors while retaining uncertainty statistics.

    OGPO reports the mean of every final-info field.  We additionally retain
    population std and SEM so comparisons do not discard the 64 episode
    distribution.  Episode length follows OGPO and is averaged over successful
    terminal episodes only; ``length_all`` keeps the all-episode diagnostic.
    """
    if not eval_metrics_list:
        return {}

    keys = set().union(*(metrics.keys() for metrics in eval_metrics_list))
    values: dict[str, torch.Tensor] = {}
    for key in keys:
        shards = []
        for metrics in eval_metrics_list:
            if key not in metrics:
                continue
            value = metrics[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            shards.append(value.detach().cpu().reshape(-1).float())
        if shards:
            values[key] = torch.cat(shards)

    if exact_episode_count is not None:
        if exact_episode_count <= 0:
            raise ValueError("exact_episode_count must be positive")
        values = {key: tensor[:exact_episode_count] for key, tensor in values.items()}

    count = max((tensor.numel() for tensor in values.values()), default=0)
    result: dict[str, float | int] = {"num_trajectories": count}
    successes = values.get("success")

    def add_stats(name: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            result[name] = 0.0
            result[f"{name}_std"] = 0.0
            result[f"{name}_sem"] = 0.0
            return
        std = tensor.std(unbiased=False)
        result[name] = float(tensor.mean())
        result[f"{name}_std"] = float(std)
        result[f"{name}_sem"] = float(std / math.sqrt(tensor.numel()))

    for key, tensor in values.items():
        if key == "length":
            add_stats("length_all", tensor)
            if successes is not None and successes.numel() == tensor.numel():
                tensor = tensor[successes > 0.5]
        add_stats(key, tensor)

    if successes is not None:
        result["success_count"] = int((successes > 0.5).sum())
    return result


class OGPOEmbodiedRunner(EmbodiedRunner):
    """Run dual ODE/SDE evaluation on OGPO's native update-step schedule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bc_finished = int(self.cfg.algorithm.bc_runner_steps) == 0
        self._actual_bc_runner_steps: int | None = 0 if self._bc_finished else None
        self._async_pending_batches = 0
        self._async_actual_collected_env_steps = 0
        self._completed_save_milestones: set[int] = set()

    def update_rollout_weights(self, source: str = "target") -> None:
        """Sync official current ODE or EMA target SDE policy weights."""
        self.actor.set_ogpo_rollout_weight_source(source).wait()
        super().update_rollout_weights()

    def _envs_per_step(self) -> int:
        return (
            int(self.cfg.env.train.total_num_envs)
            * int(self.cfg.env.train.rollout_epoch)
            * int(self.cfg.env.train.max_steps_per_rollout_epoch)
        )

    def _online_target(self) -> int:
        return int(self.cfg.algorithm.get("ogpo_online_env_steps", 500_000))

    def init_workers(self) -> None:
        """Initialize workers and restore OGPO's phase-specific progress."""
        super().init_workers()
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        state_path = os.path.join(resume_dir, "ogpo_runner_state.json")
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as state_file:
                state = json.load(state_file)
            self._bc_finished = bool(state["bc_finished"])
            actual_bc_steps = state.get("actual_bc_runner_steps")
            self._actual_bc_runner_steps = (
                int(actual_bc_steps) if actual_bc_steps is not None else None
            )
            bc_update_steps = int(state["bc_update_steps"])
            online_env_steps = int(state["online_env_steps"])
            self._async_pending_batches = int(state.get("async_pending_batches", 0))
            self._async_actual_collected_env_steps = int(
                state.get("async_actual_collected_env_steps", online_env_steps)
            )
            self._completed_save_milestones = {
                int(value) for value in state.get("completed_save_milestones", [])
            }
        else:
            # Compatibility for checkpoints produced before phase metadata was
            # introduced. Such checkpoints used the fixed 50k BC schedule.
            bc_cap = int(self.cfg.algorithm.bc_runner_steps)
            self._bc_finished = self.global_step >= bc_cap
            self._actual_bc_runner_steps = bc_cap if self._bc_finished else None
            bc_update_steps = min(self.global_step, bc_cap) * int(
                self.cfg.algorithm.bc_updates_per_step
            )
            online_env_steps = max(self.global_step - bc_cap, 0) * self._envs_per_step()
            self._async_actual_collected_env_steps = online_env_steps
            self.logger.warning(
                "OGPO checkpoint has no phase metadata; inferred the legacy "
                "fixed-50k BC schedule."
            )

        if self._bc_finished:
            online_runner_steps = math.ceil(
                self._online_target() / self._envs_per_step()
            )
            self.max_steps = self._actual_bc_runner_steps + online_runner_steps
        self.actor.restore_ogpo_training_state(
            bc_update_steps=bc_update_steps,
            online_env_steps=online_env_steps,
            bc_finished=self._bc_finished,
        ).wait()

    def _save_checkpoint(self) -> None:
        """Save the model/replay state plus OGPO's phase counters."""
        super()._save_checkpoint()
        bc_updates, online_steps, official_step = self._progress()
        checkpoint_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        state = {
            "bc_finished": self._bc_finished,
            "actual_bc_runner_steps": self._actual_bc_runner_steps,
            "bc_update_steps": bc_updates,
            "online_env_steps": online_steps,
            "official_step": official_step,
            "async_pending_batches": self._async_pending_batches,
            "async_actual_collected_env_steps": (
                self._async_actual_collected_env_steps
            ),
            "completed_save_milestones": sorted(self._completed_save_milestones),
        }
        with open(
            os.path.join(checkpoint_dir, "ogpo_runner_state.json"),
            "w",
            encoding="utf-8",
        ) as state_file:
            json.dump(state, state_file, indent=2)

    def _progress(self) -> tuple[int, int, int]:
        bc_runner_steps = int(self.cfg.algorithm.bc_runner_steps)
        bc_updates_per_step = int(self.cfg.algorithm.bc_updates_per_step)
        actual_bc_steps = (
            self._actual_bc_runner_steps
            if self._actual_bc_runner_steps is not None
            else min(self.global_step, bc_runner_steps)
        )
        bc_update_steps = actual_bc_steps * bc_updates_per_step
        online_env_steps = 0
        if self._actual_bc_runner_steps is not None:
            online_env_steps = (
                max(self.global_step - self._actual_bc_runner_steps, 0)
                * self._envs_per_step()
            )
        official_step = bc_update_steps + online_env_steps
        return bc_update_steps, online_env_steps, official_step

    def _should_evaluate(self) -> bool:
        if bool(self.cfg.runner.get("ogpo_benchmark_skip_eval", False)):
            return False
        bc_updates, online_steps, _ = self._progress()
        if not self._bc_finished:
            interval = int(self.cfg.algorithm.ogpo_eval_interval_bc_updates)
            bc_total = int(self.cfg.algorithm.bc_runner_steps) * int(
                self.cfg.algorithm.bc_updates_per_step
            )
            return bc_updates >= bc_total or (
                interval > 0 and bc_updates % interval == 0
            )
        interval = int(self.cfg.algorithm.ogpo_eval_interval_env_steps)
        previous_online_steps = max(0, online_steps - self._envs_per_step())
        return online_steps >= self._online_target() or (
            interval > 0
            and online_steps // interval > previous_online_steps // interval
        )

    def _should_log_step(self, step: int) -> bool:
        del step
        bc_updates, online_steps, _ = self._progress()
        if not self._bc_finished or online_steps == 0:
            interval = int(
                self.cfg.algorithm.get("ogpo_log_interval_bc_updates", 5_000)
            )
            return self._bc_finished or interval <= 0 or bc_updates % interval == 0
        interval = int(self.cfg.algorithm.get("ogpo_log_interval_env_steps", 5_000))
        previous_online_steps = max(0, online_steps - self._envs_per_step())
        return (
            online_steps >= self._online_target()
            or interval <= 0
            or online_steps // interval > previous_online_steps // interval
        )

    def _metric_step(self, step: int) -> int:
        """Use OGPO's combined BC-update/online-transition x-axis."""
        del step
        return self._progress()[2]

    def _finish_bc_phase(self) -> None:
        if self._bc_finished:
            return
        self._bc_finished = True
        self._actual_bc_runner_steps = self.global_step
        self.actor.finish_ogpo_bc().wait()
        online_runner_steps = math.ceil(self._online_target() / self._envs_per_step())
        self.max_steps = self.global_step + online_runner_steps
        self.logger.info(
            "OGPO BC phase ended after %d updates; online target is %d "
            "transitions (%d runner steps).",
            self._progress()[0],
            self._online_target(),
            online_runner_steps,
        )

    def _evaluate_mode(self, mode: str) -> dict:
        # ODE and SDE must see the same ordered state IDs and seeded resets.
        self.env.reset_ogpo_eval_state_sequence().wait()
        source = "current" if mode == "ode" else "target"
        self.update_rollout_weights(source=source)
        self.rollout.set_ogpo_eval_sampling_mode(mode).wait()
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel, rollout_channel=self.rollout_channel
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        return compute_ogpo_evaluate_metrics(
            [result for result in env_results if result is not None],
            exact_episode_count=(
                int(self.cfg.env.eval.rollout_epoch)
                * int(self.cfg.env.eval.total_num_envs)
            ),
        )

    def run(self) -> None:
        """Train normally or evaluate a restored checkpoint in both modes."""
        only_eval = bool(self.cfg.runner.get("only_eval", False))
        if not only_eval and bool(
            self.cfg.runner.get("use_ogpo_async_pipeline", False)
        ):
            self.run_ogpo_async_pipeline()
            return
        if not only_eval:
            super().run()
            return

        bc_updates, online_steps, official_step = self._progress()
        modes = list(self.cfg.runner.get("ogpo_eval_modes", ["ode", "sde"]))
        if not modes or any(mode not in {"ode", "sde"} for mode in modes):
            raise ValueError(f"Invalid OGPO evaluation modes: {modes}")
        eval_metrics = {}
        for mode in modes:
            with self.timer(f"eval_{mode}"):
                mode_metrics = self._evaluate_mode(mode)
            prefix = "eval" if mode == "ode" else "eval_sde"
            eval_metrics.update(
                {f"{prefix}/{key}": value for key, value in mode_metrics.items()}
            )
        eval_metrics.update(
            {
                "ogpo/bc_update_steps": bc_updates,
                "ogpo/online_env_steps": online_steps,
                "ogpo/official_step": official_step,
            }
        )
        self.metric_logger.log(data=eval_metrics, step=official_step)
        self.logger.info("OGPO checkpoint evaluation: %s", eval_metrics)
        self._finish_run()

    def _collect_async_rollout(self) -> tuple[Handle, Handle, list[dict]]:
        """Collect and ingest one rollout batch without running an update."""
        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        recv_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        )
        recv_handle.wait()
        rollout_handle.wait()
        env_handle.wait()
        actor_rollout_metrics = self.actor.compute_advantages_and_returns().wait()
        self._async_actual_collected_env_steps += self._envs_per_step()
        self._async_pending_batches += 1
        return env_handle, rollout_handle, actor_rollout_metrics

    def _log_async_state(self) -> None:
        """Log and validate the bounded one-batch pipeline state."""
        _, trained_steps, official_step = self._progress()
        envs_per_step = self._envs_per_step()
        expected_collected = trained_steps + self._async_pending_batches * envs_per_step
        if self._async_actual_collected_env_steps != expected_collected:
            raise RuntimeError(
                "OGPO async accounting mismatch: collected="
                f"{self._async_actual_collected_env_steps}, trained={trained_steps}, "
                f"pending_batches={self._async_pending_batches}, "
                f"envs_per_step={envs_per_step}"
            )
        if self._async_pending_batches not in (0, 1):
            raise RuntimeError(
                "OGPO async policy lag exceeded one batch: "
                f"pending_batches={self._async_pending_batches}"
            )
        self.metric_logger.log(
            {
                "async/actual_collected_env_steps": float(
                    self._async_actual_collected_env_steps
                ),
                "async/training_accounted_env_steps": float(trained_steps),
                "async/pending_batches": float(self._async_pending_batches),
                "async/policy_lag": float(self._async_pending_batches),
            },
            step=official_step,
        )

    def run_ogpo_async_pipeline(self) -> None:
        """Overlap online OGPO training with the next one-batch rollout.

        The rollout worker keeps the target-policy snapshot from the start of
        an iteration while the actor trains on the previously ingested batch.
        The channel therefore contains at most one newly collected batch, so
        policy lag and replay accounting remain bounded and checkpointable.
        """
        if not self._bc_finished:
            raise RuntimeError(
                "OGPO async pipeline currently requires a checkpoint with a "
                "completed BC phase."
            )
        max_policy_lag = int(self.cfg.runner.get("ogpo_async_max_policy_lag", 1))
        if max_policy_lag != 1:
            raise ValueError(
                "OGPO async pipeline currently supports only "
                "ogpo_async_max_policy_lag=1."
            )

        start_step = self.global_step
        start_time = time.time()
        envs_per_step = self._envs_per_step()
        target_steps = self._online_target()

        if self._async_pending_batches == 0:
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.update_rollout_weights(source="target")
            self._collect_async_rollout()
            self._log_async_state()

        while self._progress()[1] < target_steps:
            _, trained_steps, _ = self._progress()
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            # Do not launch a surplus rollout for the final update. This is
            # the pipeline drain that leaves the final checkpoint consistent.
            if trained_steps + envs_per_step >= target_steps:
                with self.timer("async_drain_training"):
                    actor_training_handle: Handle = self.actor.run_training()
                    actor_training_metrics = actor_training_handle.wait()
                self._async_pending_batches -= 1
                self.global_step += 1
                self._log_async_state()
                self._maybe_eval_and_checkpoint(self.global_step - 1)
                break

            with self.timer("sync_weights"):
                if self.global_step % self.weight_sync_interval == 0:
                    self.update_rollout_weights(source="target")

            with self.timer("async_overlap"):
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                actor_training_handle: Handle = self.actor.run_training()
                recv_handle: Handle = self.actor.recv_rollout_trajectories(
                    input_channel=self.actor_channel
                )

                actor_training_metrics = actor_training_handle.wait()
                self._async_pending_batches -= 1
                recv_handle.wait()
                rollout_handle.wait()
                actor_rollout_metrics = (
                    self.actor.compute_advantages_and_returns().wait()
                )
                self._async_actual_collected_env_steps += envs_per_step
                self._async_pending_batches += 1

            self.global_step += 1
            self._log_async_state()
            eval_metrics = self._maybe_eval_and_checkpoint(self.global_step - 1)
            self._log_step_metrics(
                step=self.global_step - 1,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=None,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        del step  # TensorBoard uses OGPO's official combined update-step axis.
        bc_updates, online_steps, official_step = self._progress()
        eval_metrics: dict = {}
        phase_transition = False
        if self._should_evaluate():
            with self.timer("eval_ode"):
                ode_metrics = self._evaluate_mode("ode")
            with self.timer("eval_sde"):
                sde_metrics = self._evaluate_mode("sde")
            eval_metrics.update(
                {f"eval/{key}": value for key, value in ode_metrics.items()}
            )
            eval_metrics.update(
                {f"eval_sde/{key}": value for key, value in sde_metrics.items()}
            )
            eval_metrics.update(
                {
                    "ogpo/bc_update_steps": bc_updates,
                    "ogpo/online_env_steps": online_steps,
                    "ogpo/official_step": official_step,
                }
            )
            self.metric_logger.log(data=eval_metrics, step=official_step)

            if not self._bc_finished:
                bc_limit = int(self.cfg.algorithm.bc_runner_steps) * int(
                    self.cfg.algorithm.bc_updates_per_step
                )
                early_stop_enabled = bool(
                    self.cfg.algorithm.get("ogpo_bc_early_stop", True)
                )
                threshold = float(
                    self.cfg.algorithm.get("ogpo_bc_early_stop_threshold", 0.45)
                )
                mode = str(
                    self.cfg.algorithm.get("ogpo_bc_early_stop_mode", "sde")
                ).lower()
                metric = str(
                    self.cfg.algorithm.get("ogpo_bc_early_stop_metric", "success")
                )
                prefix = "eval_sde" if mode == "sde" else "eval"
                score = float(eval_metrics.get(f"{prefix}/{metric}", float("-inf")))
                if bc_updates >= bc_limit or (
                    early_stop_enabled and score >= threshold
                ):
                    self._finish_bc_phase()
                    phase_transition = True
                    eval_metrics["ogpo/bc_early_stopped"] = float(bc_updates < bc_limit)
                    eval_metrics["ogpo/bc_stop_score"] = score
                    # Re-log transition diagnostics at the same official step.
                    self.metric_logger.log(data=eval_metrics, step=official_step)

        previous_online_steps = max(0, online_steps - self._envs_per_step())
        save_milestones = {
            int(value)
            for value in self.cfg.algorithm.get("ogpo_save_milestones", [])
            if int(value) > 0
        }
        crossed_milestones = {
            milestone
            for milestone in save_milestones
            if previous_online_steps < milestone <= online_steps
            and milestone not in getattr(self, "_completed_save_milestones", set())
        }
        save_interval = int(self.cfg.algorithm.get("ogpo_save_interval_env_steps", -1))
        crossed_save_boundary = (
            online_steps > 0
            and save_interval > 0
            and online_steps // save_interval > previous_online_steps // save_interval
        )
        should_save = (
            phase_transition
            or online_steps >= self._online_target()
            or crossed_save_boundary
            or bool(crossed_milestones)
        )
        if should_save:
            if not hasattr(self, "_completed_save_milestones"):
                self._completed_save_milestones = set()
            self._completed_save_milestones.update(crossed_milestones)
            self._save_checkpoint()
        return eval_metrics
