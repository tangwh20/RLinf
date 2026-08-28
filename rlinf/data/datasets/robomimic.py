# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Low-dimensional Robomimic datasets for OGPO behavior cloning."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.envs.robomimic.robomimic_utils import (
    TASK_LOW_DIM_KEYS,
    resolve_dataset_path,
    split_task_name,
)


class RobomimicDataset(Dataset):
    """Flatten Robomimic demonstrations into RLinf transition arrays."""

    @classmethod
    def from_path(
        cls,
        dataset_path: str | None,
        task_name: str,
        *,
        dataset_root: str | None = None,
        encoded_dataset_path: str | None = None,
        expected_observation_dim: int | None = None,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ) -> "RobomimicDataset":
        """Load an encoded cache or the original low-dimensional HDF5 format."""
        if encoded_dataset_path is not None:
            return cls._from_encoded_path(
                encoded_dataset_path,
                expected_observation_dim=expected_observation_dim,
                clip_to_eps=clip_to_eps,
                eps=eps,
            )

        task, _, _ = split_task_name(task_name)
        path = resolve_dataset_path(task_name, dataset_path, dataset_root)
        keys = TASK_LOW_DIM_KEYS[task]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        terminals: list[np.ndarray] = []
        dones_float: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []

        with h5py.File(path, "r") as dataset:
            demos = sorted(dataset["data"].keys(), key=lambda name: int(name[5:]))
            for demo in demos:
                group = dataset[f"data/{demo}"]
                obs = np.concatenate(
                    [np.asarray(group[f"obs/{key}"]) for key in keys], axis=-1
                ).astype(np.float32)
                next_obs = np.concatenate(
                    [np.asarray(group[f"next_obs/{key}"]) for key in keys], axis=-1
                ).astype(np.float32)
                action = np.asarray(group["actions"], dtype=np.float32)
                terminal = np.asarray(group["dones"], dtype=np.float32)
                boundary = terminal.copy()
                if len(boundary):
                    boundary[-1] = 1.0
                observations.append(obs)
                actions.append(action)
                rewards.append(np.asarray(group["rewards"], dtype=np.float32) - 1.0)
                terminals.append(terminal)
                dones_float.append(boundary)
                next_observations.append(next_obs)

        action_array = np.concatenate(actions)
        if clip_to_eps:
            action_array = np.clip(action_array, -1 + eps, 1 - eps)
        return cls(
            observations=np.concatenate(observations),
            actions=action_array,
            rewards=np.concatenate(rewards),
            masks=1.0 - np.concatenate(terminals),
            dones_float=np.concatenate(dones_float),
            next_observations=np.concatenate(next_observations),
        )

    @classmethod
    def _from_encoded_path(
        cls,
        path: str,
        *,
        expected_observation_dim: int | None,
        clip_to_eps: bool,
        eps: float,
    ) -> "RobomimicDataset":
        """Load a validated frozen-encoder cache produced by RLinf tooling."""
        required_keys = {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "masks",
            "dones_float",
        }
        with h5py.File(path, "r") as dataset:
            missing = required_keys.difference(dataset.keys())
            if missing:
                raise ValueError(
                    f"Encoded Robomimic dataset {path!r} is missing keys: "
                    f"{sorted(missing)}"
                )
            if int(dataset.attrs.get("complete", 0)) != 1:
                raise ValueError(
                    f"Encoded Robomimic dataset {path!r} is not marked complete."
                )
            observations = np.asarray(dataset["observations"], dtype=np.float32)
            next_observations = np.asarray(
                dataset["next_observations"], dtype=np.float32
            )
            actions = np.asarray(dataset["actions"], dtype=np.float32)
            rewards = np.asarray(dataset["rewards"], dtype=np.float32)
            masks = np.asarray(dataset["masks"], dtype=np.float32)
            dones_float = np.asarray(dataset["dones_float"], dtype=np.float32)

        size = observations.shape[0]
        fields = {
            "next_observations": next_observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "dones_float": dones_float,
        }
        mismatched = {
            name: value.shape[0]
            for name, value in fields.items()
            if value.shape[0] != size
        }
        if mismatched:
            raise ValueError(
                f"Encoded Robomimic fields do not match observations length {size}: "
                f"{mismatched}"
            )
        if observations.ndim != 2 or next_observations.shape != observations.shape:
            raise ValueError(
                "Encoded observations and next_observations must have identical "
                f"[N, D] shapes, got {observations.shape} and "
                f"{next_observations.shape}."
            )
        if (
            expected_observation_dim is not None
            and observations.shape[-1] != expected_observation_dim
        ):
            raise ValueError(
                f"Expected encoded observation dim {expected_observation_dim}, "
                f"got {observations.shape[-1]}."
            )
        if (
            not np.isfinite(observations).all()
            or not np.isfinite(next_observations).all()
        ):
            raise ValueError("Encoded Robomimic observations contain NaN or Inf.")
        if clip_to_eps:
            actions = np.clip(actions, -1 + eps, 1 - eps)
        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
        )

    def __init__(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = len(observations)
        self._torch_cache: dict[str, torch.Tensor] | None = None

    def _ensure_torch_cache(self) -> None:
        if self._torch_cache is None:
            self._torch_cache = {
                "observations": torch.from_numpy(self.observations).float(),
                "actions": torch.from_numpy(self.actions).float(),
                "rewards": torch.from_numpy(self.rewards).float(),
                "masks": torch.from_numpy(self.masks).float(),
                "next_observations": torch.from_numpy(self.next_observations).float(),
            }

    def __len__(self) -> int:
        return int(self.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_torch_cache()
        assert self._torch_cache is not None
        return {key: value[idx] for key, value in self._torch_cache.items()}

    def get_obs_action_dims(self) -> tuple[int, int]:
        """Return flattened observation and action dimensions."""
        return int(self.observations.shape[-1]), int(self.actions.shape[-1])

    def get_dataset_size(self) -> int:
        """Return the transition count."""
        return int(self.size)


def build_robomimic_dataset_from_cfg(cfg: Any) -> RobomimicDataset:
    """Build a Robomimic dataset from RLinf's ``data`` configuration."""
    return RobomimicDataset.from_path(
        dataset_path=cfg.data.get("dataset_path", None),
        dataset_root=cfg.data.get("dataset_root", None),
        encoded_dataset_path=cfg.data.get("encoded_dataset_path", None),
        expected_observation_dim=int(cfg.actor.model.obs_dim),
        task_name=str(cfg.data.task_name),
    )
