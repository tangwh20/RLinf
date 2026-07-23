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

import sys
import types
from pathlib import Path

import numpy as np

from rlinf.envs.libero_safety.metrics import (
    aggregate_constraint_costs,
    classify_safety_success,
)


def test_aggregate_constraint_costs_uses_any_violated_predicate():
    costs = aggregate_constraint_costs(
        [
            {"And": 0},
            {"contact": 0, "force": 1},
            {},
            None,
        ]
    )

    np.testing.assert_array_equal(costs, np.array([0, 1, 0, 0], dtype=np.float32))


def test_classify_safety_success_reports_joint_outcomes():
    safe_success, unsafe_success = classify_safety_success(
        np.array([True, True, False, False]),
        np.array([False, True, False, True]),
    )

    np.testing.assert_array_equal(safe_success, [True, False, False, False])
    np.testing.assert_array_equal(unsafe_success, [False, True, False, False])


def test_configure_safety_purges_standard_namespace_package(monkeypatch, tmp_path):
    from rlinf.envs import _configure_libero_safety

    monkeypatch.setattr(sys, "path", sys.path.copy())
    for variable in (
        "LIBERO_CONFIG_PATH",
        "LIBERO_SAFETY_REPO_PATH",
        "LIBERO_TYPE",
    ):
        monkeypatch.delenv(variable, raising=False)

    repo_path = tmp_path / "libero_safety"
    task_map = repo_path / "libero" / "libero" / "benchmark" / "vla_safety_task_map.py"
    task_map.parent.mkdir(parents=True)
    task_map.touch()

    standard_libero = types.ModuleType("libero")
    standard_libero.__file__ = None
    standard_benchmark = types.ModuleType("libero.libero.benchmark")
    standard_benchmark.__file__ = "/opt/standard/libero/benchmark/__init__.py"
    monkeypatch.setitem(sys.modules, "libero", standard_libero)
    monkeypatch.setitem(
        sys.modules, "libero.libero.benchmark", standard_benchmark
    )

    _configure_libero_safety({"repo_path": str(repo_path)})

    assert list(sys.modules["libero"].__path__) == [str(repo_path / "libero")]
    assert "libero.libero.benchmark" not in sys.modules
    assert Path(sys.path[0]) == repo_path


def test_safety_benchmark_is_resolved_after_standard_libero_import(monkeypatch):
    standard_libero = types.ModuleType("libero")
    standard_libero.__path__ = []
    standard_core = types.ModuleType("libero.libero")
    standard_core.__path__ = []
    standard_benchmark = types.ModuleType("libero.libero.benchmark")

    class StandardBenchmark:
        pass

    standard_benchmark.Benchmark = StandardBenchmark
    standard_benchmark.BENCHMARK_MAPPING = {}
    standard_libero.libero = standard_core
    standard_core.benchmark = standard_benchmark
    monkeypatch.setitem(sys.modules, "libero", standard_libero)
    monkeypatch.setitem(sys.modules, "libero.libero", standard_core)
    monkeypatch.setitem(
        sys.modules, "libero.libero.benchmark", standard_benchmark
    )

    from rlinf.envs.libero import utils

    safety_benchmark = types.ModuleType("libero.libero.benchmark")

    class SafetyBenchmark:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

        def get_num_tasks(self):
            return self.n_tasks

        def get_task(self, task_id):
            return self.tasks[task_id]

    def make_suite(name):
        class Suite(SafetyBenchmark):
            def __init__(self, task_order_index=0):
                super().__init__(task_order_index)
                self.tasks = [f"{name}_task"]
                self.n_tasks = len(self.tasks)

        return Suite

    safety_benchmark.Benchmark = SafetyBenchmark
    safety_benchmark.BENCHMARK_MAPPING = {
        name: make_suite(name)
        for name in (
            "affordance",
            "human_safety",
            "obstacle_avoidance",
            "obstacle_avoidance_human",
        )
    }
    safety_benchmark.get_benchmark = safety_benchmark.BENCHMARK_MAPPING.__getitem__

    monkeypatch.setenv("LIBERO_TYPE", "safety")
    monkeypatch.setitem(
        sys.modules, "libero.libero.benchmark", safety_benchmark
    )

    suite_cls = utils.get_benchmark_overridden("libero_safety")
    suite = suite_cls()

    assert suite.get_num_tasks() == 4
    assert suite.get_task(0) == "affordance_task"
    assert safety_benchmark.BENCHMARK_MAPPING["libero_safety"] is suite_cls
