# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Low-dimensional Robomimic environment helpers adapted from OGPO_public."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gym
import h5py
import numpy as np
from gym.spaces import Box

from rlinf.models.embodiment.observation_encoders.paligemma_client import (
    PaliGemmaEncoderClient,
)

TASK_LOW_DIM_KEYS = {
    "lift": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "can": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "square": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "transport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "tool_hang": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
}

TASK_PROPRIO_KEYS = {
    task: ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    for task in ("lift", "can", "square", "tool_hang")
}

MAX_EPISODE_LENGTH = {
    "lift": 300,
    "can": 300,
    "square": 400,
    "transport": 800,
    "tool_hang": 1000,
}


def split_task_name(task_name: str) -> tuple[str, str, str]:
    """Split ``task-dataset-observation`` Robomimic task names."""
    try:
        task, dataset_type, observation_type = task_name.split("-")
    except ValueError as exc:
        raise ValueError(
            "Robomimic task names must have the form task-dataset-observation, "
            f"got {task_name!r}."
        ) from exc
    if task not in TASK_LOW_DIM_KEYS:
        raise ValueError(f"Unsupported Robomimic task {task!r}.")
    if observation_type not in {"low_dim", "image"}:
        raise ValueError(
            f"Unsupported Robomimic observation type {observation_type!r}."
        )
    return task, dataset_type, observation_type


def resolve_dataset_path(
    task_name: str,
    dataset_path: str | os.PathLike[str] | None = None,
    dataset_root: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a Robomimic dataset path using OGPO_public's directory layout."""
    if dataset_path is not None:
        return str(dataset_path)
    task, dataset_type, observation_type = split_task_name(task_name)
    root = dataset_root or os.environ.get("ROBOMIMIC_DATASET_ROOT")
    if root is None:
        root = Path.home() / ".robomimic"
    filename = "image_v15.hdf5" if observation_type == "image" else "low_dim_v15.hdf5"
    return str(Path(root) / task / dataset_type / filename)


class RobomimicPaliGemmaWrapper(gym.Env):
    """Expose frozen PaliGemma features as a flat Robomimic observation."""

    def __init__(
        self,
        env: Any,
        *,
        proprio_keys: list[str],
        privileged_state_keys: list[str],
        append_privileged_state: bool,
        image_key: str,
        max_episode_length: int,
        encoder_host: str,
        encoder_port: int,
        post_success_steps: int = 0,
    ) -> None:
        self.env = env
        self.proprio_keys = proprio_keys
        self.privileged_state_keys = privileged_state_keys
        self.append_privileged_state = bool(append_privileged_state)
        self.image_key = image_key
        self.max_episode_length = int(max_episode_length)
        self.post_success_steps = int(post_success_steps)
        self.encoder = PaliGemmaEncoderClient(encoder_host, encoder_port)
        health = self.encoder.health()
        if int(health["output_dim"]) != 2057:
            raise RuntimeError(f"Unexpected encoder health response: {health}")
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.t_succ: int | None = None

        action_low = np.full(env.action_dimension, -1.0, dtype=np.float32)
        action_high = np.full(env.action_dimension, 1.0, dtype=np.float32)
        self.action_space = Box(action_low, action_high, dtype=np.float32)
        raw_obs = self.env.get_observation()
        self.privileged_state_dim = sum(
            int(np.asarray(raw_obs[key]).size) for key in self.privileged_state_keys
        )
        observation_dim = 2057 + (
            self.privileged_state_dim if self.append_privileged_state else 0
        )
        self.observation_space = Box(
            low=np.full((observation_dim,), -np.inf, dtype=np.float32),
            high=np.full((observation_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _encode(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        proprio = np.concatenate(
            [np.asarray(raw_obs[key], dtype=np.float32) for key in self.proprio_keys]
        )
        image = np.asarray(raw_obs[self.image_key])
        actor_obs = self.encoder.encode(proprio[None], image[None])[0]
        if not self.append_privileged_state:
            return actor_obs
        critic_obs = np.concatenate(
            [
                np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
                for key in self.privileged_state_keys
            ]
        )
        # Keep RLinf's flat observation transport unchanged. The policy splits
        # the leading actor observation from the trailing privileged critic
        # observation. Offline BC remains the original 2057-D actor input.
        return np.concatenate([actor_obs, critic_obs]).astype(np.float32)

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed the global RNG used by the original Robomimic environment."""
        np.random.seed(seed)
        return [seed]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and encode the first observation of an episode."""
        del options, kwargs
        self.t = 0
        self.t_succ = None
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes += 1
        if seed is not None:
            self.seed(seed)
        raw_obs = self.env.reset()
        return self._encode(raw_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step Robomimic and encode the resulting image observation."""
        raw_obs, raw_reward, done, info = self.env.step(action)
        reward = float(raw_reward) - 1.0
        is_success_step = reward > -0.5
        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            info["success"] = 1
            steps_since_success = self.t - self.t_succ
            if is_success_step and steps_since_success <= self.post_success_steps:
                reward += 5.0
            if steps_since_success >= self.post_success_steps or not is_success_step:
                done = True

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1
        terminated = bool(done)
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info["return"] = self.episode_return
            info["length"] = self.episode_length
        return (
            self._encode(raw_obs),
            reward,
            terminated,
            bool(truncated and not terminated),
            info,
        )

    def render(self, mode: str = "rgb_array", **kwargs: Any) -> np.ndarray:
        """Render the agent-view camera."""
        del kwargs
        return self.env.render(
            mode=mode, height=256, width=256, camera_name="agentview"
        )

    def close(self) -> None:
        """Close both encoder connection and Robomimic environment."""
        self.encoder.close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class RobomimicLowdimWrapper(gym.Env):
    """Expose OGPO_public's low-dimensional Robomimic Gymnasium contract."""

    def __init__(
        self,
        env: Any,
        low_dim_keys: list[str],
        max_episode_length: int,
        post_success_steps: int = 0,
        render_hw: tuple[int, int] = (256, 256),
        render_camera_name: str = "agentview",
    ) -> None:
        self.env = env
        self.low_dim_keys = low_dim_keys
        self.max_episode_length = int(max_episode_length)
        self.post_success_steps = int(post_success_steps)
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.t_succ: int | None = None

        action_low = np.full(env.action_dimension, -1.0, dtype=np.float32)
        action_high = np.full(env.action_dimension, 1.0, dtype=np.float32)
        self.action_space = Box(action_low, action_high, dtype=np.float32)
        obs = self.get_observation().astype(np.float32)
        self.observation_space = Box(
            low=np.full_like(obs, -np.inf),
            high=np.full_like(obs, np.inf),
            dtype=np.float32,
        )

    def get_observation(self) -> np.ndarray:
        """Return the flattened low-dimensional observation."""
        raw_obs = self.env.get_observation()
        return np.concatenate([raw_obs[key] for key in self.low_dim_keys]).astype(
            np.float32
        )

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed the global RNG used by the original Robomimic environment."""
        np.random.seed(seed)
        return [seed]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset an episode using OGPO_public's seeded-reset behavior."""
        del options, kwargs
        self.t = 0
        self.t_succ = None
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes += 1
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        return self.get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step once with OGPO_public reward and success semantics."""
        raw_obs, raw_reward, done, info = self.env.step(action)
        reward = float(raw_reward) - 1.0
        obs = np.concatenate([raw_obs[key] for key in self.low_dim_keys]).astype(
            np.float32
        )
        is_success_step = reward > -0.5
        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            info["success"] = 1
            steps_since_success = self.t - self.t_succ
            if is_success_step and steps_since_success <= self.post_success_steps:
                reward += 5.0
            if steps_since_success >= self.post_success_steps or not is_success_step:
                done = True

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1
        terminated = bool(done)
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info["return"] = self.episode_return
            info["length"] = self.episode_length
        return obs, reward, terminated, bool(truncated and not terminated), info

    def render(self, mode: str = "rgb_array", **kwargs: Any) -> np.ndarray:
        """Render the configured Robosuite camera."""
        del kwargs
        height, width = self.render_hw
        return self.env.render(
            mode=mode,
            height=height,
            width=width,
            camera_name=self.render_camera_name,
        )

    def close(self) -> None:
        """Close the underlying Robomimic environment when supported."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def make_env(
    task_name: str,
    *,
    dataset_path: str | os.PathLike[str] | None = None,
    dataset_root: str | os.PathLike[str] | None = None,
    seed: int = 0,
    post_success_steps: int = 0,
    encoder_host: str = "127.0.0.1",
    encoder_port: int = 29571,
    append_privileged_state: bool = False,
) -> RobomimicLowdimWrapper | RobomimicPaliGemmaWrapper:
    """Create a low-dimensional or frozen-PaliGemma Robomimic environment."""
    from robomimic.utils import env_utils as env_utils
    from robomimic.utils import obs_utils as obs_utils

    task, _, observation_type = split_task_name(task_name)
    path = resolve_dataset_path(task_name, dataset_path, dataset_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Robomimic dataset not found: {path}")
    low_dim_keys = TASK_LOW_DIM_KEYS[task]
    if observation_type == "image":
        proprio_keys = TASK_PROPRIO_KEYS[task]
        obs_utils.initialize_obs_modality_mapping_from_dict(
            {"low_dim": low_dim_keys, "rgb": ["agentview_image"]}
        )
    else:
        obs_utils.initialize_obs_modality_mapping_from_dict({"low_dim": low_dim_keys})
    with h5py.File(path, "r") as dataset:
        metadata = json.loads(dataset["data"].attrs["env_args"])
    metadata["env_kwargs"].pop("env_lang", None)
    env_utils.set_env_specific_obs_processing(env_meta=metadata)
    env = env_utils.create_env_from_metadata(
        env_meta=metadata,
        render=False,
        render_offscreen=False,
        use_image_obs=observation_type == "image",
    )
    if observation_type == "image":
        wrapped = RobomimicPaliGemmaWrapper(
            env,
            proprio_keys=proprio_keys,
            privileged_state_keys=low_dim_keys,
            append_privileged_state=append_privileged_state,
            image_key="agentview_image",
            max_episode_length=MAX_EPISODE_LENGTH[task],
            encoder_host=encoder_host,
            encoder_port=encoder_port,
            post_success_steps=post_success_steps,
        )
    else:
        wrapped = RobomimicLowdimWrapper(
            env,
            low_dim_keys=low_dim_keys,
            max_episode_length=MAX_EPISODE_LENGTH[task],
            post_success_steps=post_success_steps,
        )
    wrapped.seed(seed)
    wrapped.env.hard_reset = False
    return wrapped


def inspect_dataset(path: str) -> tuple[int, int]:
    """Return the demonstration and transition counts for diagnostics."""
    with h5py.File(path, "r") as dataset:
        demos = list(dataset["data"].keys())
        transitions = sum(dataset[f"data/{demo}/actions"].shape[0] for demo in demos)
    return len(demos), int(transitions)
