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

import os
import sys
import types
from pathlib import Path

import yaml

from rlinf.envs.libero.controller_profile import controller_profile
from rlinf.envs.libero.standard_config import configure_standard_libero


def test_standard_controller_profile_is_scoped(monkeypatch):
    robosuite = types.ModuleType("robosuite")
    native_config = {
        "type": "OSC_POSE",
        "output_max": [2] * 6,
        "output_min": [-2] * 6,
        "kp": 750,
        "kp_limits": [0, 1000],
    }

    def native_loader(*_args, **_kwargs):
        return native_config

    robosuite.load_controller_config = native_loader
    monkeypatch.setitem(sys.modules, "robosuite", robosuite)

    with controller_profile("standard"):
        standard = robosuite.load_controller_config(
            default_controller="OSC_POSE"
        )
        assert standard["output_max"] == [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        assert standard["kp"] == 150
        assert native_config["output_max"] == [2] * 6

    assert robosuite.load_controller_config is native_loader


def test_configure_standard_libero_purges_safety_package(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for variable in ("LIBERO_CONFIG_PATH", "LIBERO_REPO_PATH", "LIBERO_TYPE"):
        monkeypatch.delenv(variable, raising=False)

    repo_path = tmp_path / "libero"
    benchmark_init = repo_path / "libero" / "libero" / "benchmark" / "__init__.py"
    benchmark_init.parent.mkdir(parents=True)
    benchmark_init.touch()

    safety_benchmark = types.ModuleType("libero.libero.benchmark")
    safety_benchmark.__file__ = "/opt/libero_safety/benchmark/__init__.py"
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", safety_benchmark)

    configured = configure_standard_libero({"repo_path": str(repo_path)})

    assert configured == repo_path.resolve()
    assert "libero.libero.benchmark" not in sys.modules
    assert Path(sys.path[0]) == repo_path
    assert sys.modules.get("libero") is None
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"])
    with (config_path / "config.yaml").open() as config_file:
        paths = yaml.safe_load(config_file)
    assert Path(paths["benchmark_root"]) == repo_path / "libero" / "libero"
