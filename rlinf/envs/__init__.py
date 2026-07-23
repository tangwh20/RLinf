# Copyright 2025 The RLinf Authors.
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

import os
import sys
import tempfile
import types
from enum import Enum
from pathlib import Path

import yaml


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    MANISKILL_RLT = "maniskill_rlt"
    LIBERO = "libero"
    LIBERO_SAFETY = "libero_safety"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"
    WANWM = "wan_wm"
    GENESIS = "genesis"
    EMBODICHAIN = "embodichain"
    ROBOVERSE = "roboverse"
    D4RL = "d4rl"
    POLARIS = "polaris"


def _configure_libero_safety(env_cfg) -> Path:
    """Route the import-compatible ``libero`` package to LIBERO-Safety."""
    configured_path = env_cfg.get("repo_path", None) if env_cfg is not None else None
    project_root = Path(__file__).resolve().parents[2]
    default_candidates = (
        project_root.parent / "LIBERO-Safety",
        project_root / ".venv" / "libero_safety",
    )
    requested_path = configured_path or os.environ.get("LIBERO_SAFETY_REPO_PATH")
    repo_path = (
        Path(requested_path).expanduser().resolve()
        if requested_path
        else next(
            (path.resolve() for path in default_candidates if path.is_dir()),
            default_candidates[0].resolve(),
        )
    )
    core_path = repo_path / "libero" / "libero"
    if not (core_path / "benchmark" / "vla_safety_task_map.py").is_file():
        raise FileNotFoundError(
            "LIBERO-Safety source was not found. Set env.*.repo_path or "
            f"LIBERO_SAFETY_REPO_PATH to its repository root (resolved: {repo_path})."
        )

    config_dir = Path(tempfile.gettempdir()) / f"rlinf_libero_safety_{os.getpid()}"
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
    os.environ["LIBERO_SAFETY_REPO_PATH"] = str(repo_path)
    os.environ["LIBERO_TYPE"] = "safety"
    repo_str = str(repo_path)
    sys.path[:] = [path for path in sys.path if path != repo_str]
    sys.path.insert(0, repo_str)

    loaded_benchmark = sys.modules.get("libero.libero.benchmark")
    loaded_file = (
        getattr(loaded_benchmark, "__file__", "") if loaded_benchmark else ""
    )
    benchmark_is_safety = bool(
        loaded_file and Path(loaded_file).resolve().is_relative_to(repo_path)
    )
    if not benchmark_is_safety:
        for module_name in list(sys.modules):
            if module_name == "libero" or module_name.startswith("libero."):
                del sys.modules[module_name]

        # LIBERO-Safety's outer ``libero/`` directory has no __init__.py. A
        # separately installed standard LIBERO is therefore preferred as a
        # regular package over this namespace portion, even when repo_path is
        # first on sys.path. Pin the namespace explicitly to the Safety tree.
        safety_namespace = types.ModuleType("libero")
        safety_namespace.__file__ = None
        safety_namespace.__package__ = "libero"
        safety_namespace.__path__ = [str(repo_path / "libero")]
        sys.modules["libero"] = safety_namespace
    return repo_path


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if env_cfg.get("enable_offload", False):
            from rlinf.envs.maniskill.maniskill_offload_env import ManiskillOffloadEnv

            return ManiskillOffloadEnv
        else:
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

            return ManiskillEnv
    elif env_type == SupportedEnvType.MANISKILL_RLT:
        from rlinf.envs.maniskill.maniskill_rlt_env import ManiskillRLTEnv

        return ManiskillRLTEnv
    elif env_type == SupportedEnvType.LIBERO:
        from rlinf.envs.libero.standard_config import configure_standard_libero

        configure_standard_libero(env_cfg)
        from rlinf.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == SupportedEnvType.LIBERO_SAFETY:
        _configure_libero_safety(env_cfg)
        from rlinf.envs.libero_safety.libero_safety_env import LiberoSafetyEnv

        return LiberoSafetyEnv
    elif env_type == SupportedEnvType.ROBOTWIN:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    elif env_type == SupportedEnvType.ISAACLAB:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        assert task_id in REGISTER_ISAACLAB_ENVS, (
            f"Task type {task_id} has not been registered! "
            f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
        )
        return REGISTER_ISAACLAB_ENVS[task_id]
    elif env_type == SupportedEnvType.METAWORLD:
        from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == SupportedEnvType.BEHAVIOR:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == SupportedEnvType.CALVIN:
        from rlinf.envs.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == SupportedEnvType.ROBOCASA:
        from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == SupportedEnvType.REALWORLD:
        from rlinf.envs.realworld import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.HABITAT:
        from rlinf.envs.habitat.habitat_env import HabitatEnv

        return HabitatEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    elif env_type == SupportedEnvType.GENESIS:
        from rlinf.envs.genesis.genesis_env import GenesisEnv

        return GenesisEnv
    elif env_type == SupportedEnvType.OPENSORAWM:
        from rlinf.envs.world_model.world_model_opensora_env import OpenSoraEnv

        return OpenSoraEnv
    elif env_type == SupportedEnvType.WANWM:
        from rlinf.envs.world_model.world_model_wan_env import WanEnv

        return WanEnv
    elif env_type == SupportedEnvType.EMBODICHAIN:
        from rlinf.envs.embodichain.embodichain_env import EmbodiChainEnv

        return EmbodiChainEnv
    elif env_type == SupportedEnvType.ROBOVERSE:
        from rlinf.envs.roboverse.roboverse_env import RoboVerseEnv

        return RoboVerseEnv
    elif env_type == SupportedEnvType.D4RL:
        from rlinf.envs.d4rl.d4rl_env import D4RLEnv

        return D4RLEnv
    elif env_type == SupportedEnvType.POLARIS:
        from rlinf.envs.polaris.polaris_env import PolarisEnv

        return PolarisEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
