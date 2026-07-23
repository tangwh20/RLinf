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

"""Safety-cost conversion helpers."""

from collections.abc import Mapping, Sequence

import numpy as np


def aggregate_constraint_costs(cost_infos: Sequence[Mapping | None]) -> np.ndarray:
    """Convert LIBERO-Safety predicate dictionaries to one cost per environment."""
    costs = np.zeros(len(cost_infos), dtype=np.float32)
    for env_id, predicates in enumerate(cost_infos):
        if not predicates:
            continue
        values = [float(value) for value in predicates.values()]
        costs[env_id] = max(values, default=0.0)
    return costs


def classify_safety_success(
    success_once: np.ndarray, safety_violation_once: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split task successes into safe and unsafe episode-level successes."""
    success = np.asarray(success_once, dtype=bool)
    violation = np.asarray(safety_violation_once, dtype=bool)
    if success.shape != violation.shape:
        raise ValueError(
            "success_once and safety_violation_once must have the same shape, "
            f"got {success.shape} and {violation.shape}"
        )
    return success & ~violation, success & violation
