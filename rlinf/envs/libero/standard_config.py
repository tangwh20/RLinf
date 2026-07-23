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

"""Import routing for a standard LIBERO tree in a mixed installation."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml


def configure_standard_libero(env_cfg=None) -> Path:
    """Route ``libero`` imports and assets to the standard LIBERO checkout."""
    configured_path = env_cfg.get("repo_path", None) if env_cfg is not None else None
    project_root = Path(__file__).resolve().parents[3]
    requested_path = configured_path or os.environ.get("LIBERO_REPO_PATH")
    repo_path = (
        Path(requested_path).expanduser().resolve()
        if requested_path
        else (project_root / ".venv" / "libero").resolve()
    )
    core_path = repo_path / "libero" / "libero"
    if not (core_path / "benchmark" / "__init__.py").is_file():
        raise FileNotFoundError(
            "Standard LIBERO source was not found. Set env.*.repo_path or "
            f"LIBERO_REPO_PATH to its repository root (resolved: {repo_path})."
        )

    config_dir = Path(tempfile.gettempdir()) / f"rlinf_libero_{os.getpid()}"
    config_dir.mkdir(parents=True, exist_ok=True)
    path_config = {
        "assets": str(core_path / "assets"),
        "bddl_files": str(core_path / "bddl_files"),
        "benchmark_root": str(core_path),
        "datasets": str(repo_path / "libero" / "datasets"),
        "init_states": str(core_path / "init_files"),
    }
    with (config_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(path_config, config_file)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["LIBERO_REPO_PATH"] = str(repo_path)
    os.environ["LIBERO_TYPE"] = "standard"
    repo_str = str(repo_path)
    sys.path[:] = [path for path in sys.path if path != repo_str]
    sys.path.insert(0, repo_str)

    loaded_benchmark = sys.modules.get("libero.libero.benchmark")
    loaded_file = (
        getattr(loaded_benchmark, "__file__", "") if loaded_benchmark else ""
    )
    benchmark_is_standard = bool(
        loaded_file and Path(loaded_file).resolve().is_relative_to(repo_path)
    )
    if not benchmark_is_standard:
        for module_name in list(sys.modules):
            if module_name == "libero" or module_name.startswith("libero."):
                del sys.modules[module_name]
    return repo_path
