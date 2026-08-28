# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""RLinf vector environment adapter for low-dimensional Robomimic tasks."""

from __future__ import annotations

from typing import Any

import gym

from rlinf.envs.d4rl.d4rl_env import D4RLEnv, _cfg_get
from rlinf.envs.robomimic.robomimic_utils import make_env

__all__ = ["RobomimicEnv"]


class RobomimicEnv(D4RLEnv):
    """Use Robomimic tasks with RLinf's D4RL-compatible chunk API."""

    @staticmethod
    def _make_single_env(task_name: str, render_mode: str | None, cfg: Any) -> gym.Env:
        del render_mode
        return make_env(
            task_name,
            dataset_path=_cfg_get(cfg, "dataset_path", None),
            dataset_root=_cfg_get(cfg, "dataset_root", None),
            seed=int(_cfg_get(cfg, "seed", 0)),
            post_success_steps=int(_cfg_get(cfg, "post_success_steps", 0)),
            encoder_host=str(_cfg_get(cfg, "encoder_host", "127.0.0.1")),
            encoder_port=int(_cfg_get(cfg, "encoder_port", 29571)),
            append_privileged_state=bool(
                _cfg_get(cfg, "append_privileged_state", False)
            ),
        )

    @staticmethod
    def _build_score_env(task_name: str) -> None:
        del task_name
        return None
