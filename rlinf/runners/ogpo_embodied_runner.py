# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OGPO runner with official Adroit evaluation cadence and metric semantics."""

import json
import math
import os

import torch

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle


def compute_ogpo_evaluate_metrics(eval_metrics_list: list[dict]) -> dict:
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
        self._bc_finished = False
        self._actual_bc_runner_steps: int | None = None

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
            [result for result in env_results if result is not None]
        )

    def run(self) -> None:
        """Train normally or evaluate a restored checkpoint in both modes."""
        if not bool(self.cfg.runner.get("only_eval", False)):
            super().run()
            return

        bc_updates, online_steps, official_step = self._progress()
        with self.timer("eval_ode"):
            ode_metrics = self._evaluate_mode("ode")
        with self.timer("eval_sde"):
            sde_metrics = self._evaluate_mode("sde")
        eval_metrics = {f"eval/{key}": value for key, value in ode_metrics.items()}
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
        self.logger.info("OGPO checkpoint evaluation: %s", eval_metrics)
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

        save_interval = int(self.cfg.algorithm.get("ogpo_save_interval_env_steps", -1))
        previous_online_steps = max(0, online_steps - self._envs_per_step())
        crossed_save_boundary = (
            online_steps > 0
            and save_interval > 0
            and online_steps // save_interval > previous_online_steps // save_interval
        )
        should_save = (
            phase_transition
            or online_steps >= self._online_target()
            or crossed_save_boundary
        )
        if should_save:
            self._save_checkpoint()
        return eval_metrics
