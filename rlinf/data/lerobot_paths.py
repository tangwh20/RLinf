# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve LeRobot dataset paths without relying on a global HF_LEROBOT_HOME override."""

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HOME
from omegaconf import DictConfig, ListConfig

_DEFAULT_LEROBOT_HOME = Path(HF_HOME) / "lerobot"


def default_hf_lerobot_home() -> Path:
    return Path(os.getenv("HF_LEROBOT_HOME", _DEFAULT_LEROBOT_HOME)).expanduser()


def ensure_hf_datasets_list_feature_compat() -> None:
    """Teach older Hugging Face Datasets releases to read List metadata.

    LeRobot v2.1 datasets written by recent Datasets releases serialize
    fixed-size list columns with "_type: List". Datasets 3.x represents the
    same Arrow type as Sequence and otherwise fails before reading parquet rows.
    """
    from datasets import Features, Sequence
    from datasets.features.features import register_feature

    probe = {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 1,
        "_type": "List",
    }
    try:
        Features.from_dict({"_list_compat_probe": probe})
    except ValueError as exc:
        if "Feature type 'List' not found" not in str(exc):
            raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            register_feature(Sequence, "List")


def suppress_torchcodec_info_logs() -> None:
    """Keep FFmpeg errors while suppressing per-decoder codec info messages."""
    try:
        import torchcodec.decoders  # noqa: F401

        libavutil = ctypes.CDLL("libavutil.so")
        av_log_set_level = libavutil.av_log_set_level
        av_log_set_level.argtypes = [ctypes.c_int]
        av_log_set_level.restype = None
        av_log_set_level(16)  # AV_LOG_ERROR
    except (AttributeError, ImportError, OSError):
        # TorchCodec is optional and may use a statically linked FFmpeg build.
        pass


def quiet_openpi_data_worker_init(worker_id: int) -> None:
    """Initialize a spawned OpenPI data worker with quiet FFmpeg logging."""
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    suppress_torchcodec_info_logs()


def install_quiet_openpi_data_worker_init(data_loader_module: Any) -> None:
    """Install the spawn-safe worker initializer used by OpenPI's DataLoader."""
    data_loader_module._worker_init_fn = quiet_openpi_data_worker_init


def _mapping_get(mapping: Any, key: str, default: Any = None) -> Any:
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return default


def resolve_lerobot_repo_id(data_paths: Any) -> str | None:
    """Extract a LeRobot repo id or local dataset path from ``data.train_data_paths``."""
    if data_paths is None:
        return None
    if isinstance(data_paths, (str, Path)):
        return str(data_paths)
    if isinstance(data_paths, (dict, DictConfig)):
        path = _mapping_get(data_paths, "dataset_path") or _mapping_get(
            data_paths, "data_path"
        )
        return str(path) if path is not None else None
    if isinstance(data_paths, (list, tuple, ListConfig)):
        if len(data_paths) == 0:
            return None
        return resolve_lerobot_repo_id(data_paths[0])
    return str(data_paths)


def resolve_lerobot_dataset_root(data_path: str) -> Path:
    """Resolve the on-disk LeRobot dataset root for a path or Hugging Face repo id."""
    path = Path(data_path).expanduser()
    if (path / "meta" / "info.json").is_file():
        return path.resolve()

    cached = default_hf_lerobot_home() / data_path
    if (cached / "meta" / "info.json").is_file():
        return cached.resolve()

    return path.resolve()
