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

"""RLinf wrapper for the LIBERO-Safety simulator."""

import numpy as np

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.libero_safety.metrics import (
    aggregate_constraint_costs,
    classify_safety_success,
)
from rlinf.envs.utils import to_tensor
from rlinf.utils.logging import get_logger

logger = get_logger()


class LiberoSafetyEnv(LiberoEnv):
    """LIBERO vector environment with constraint-cost rewards and metrics."""

    def __init__(self, *args, **kwargs):
        self.current_safety_cost = np.array([], dtype=np.float32)
        super().__init__(*args, **kwargs)
        self.safety_cost_coef = float(self.cfg.get("safety_cost_coef", 1.0))

    def _log_evaluation_mode(self):
        level = self.cfg.get("safety_level", "all")
        logger.info(
            "Evaluation Mode: LIBERO-Safety | Suite: %s | Level: %s",
            self.cfg.task_suite_name,
            level,
        )

    @property
    def info_logging_keys(self):
        return ["safety_cost"]

    def _init_metrics(self):
        super()._init_metrics()
        self.safety_cost_sum = np.zeros(self.num_envs, dtype=np.float32)
        self.safety_violation_once = np.zeros(self.num_envs, dtype=bool)
        self.current_safety_cost = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if env_idx is None:
            self.safety_cost_sum[:] = 0.0
            self.safety_violation_once[:] = False
            self.current_safety_cost[:] = 0.0
        else:
            self.safety_cost_sum[env_idx] = 0.0
            self.safety_violation_once[env_idx] = False
            self.current_safety_cost[env_idx] = 0.0

    def _process_step_infos(self, infos):
        predicate_costs = infos.get("cost", [{} for _ in range(self.num_envs)])
        self.current_safety_cost = aggregate_constraint_costs(predicate_costs)
        scalar_cost = to_tensor(self.current_safety_cost)
        infos["cost"] = scalar_cost
        infos["safety_cost"] = scalar_cost
        return infos

    def _record_metrics(self, step_reward, terminations, infos):
        self.safety_cost_sum += self.current_safety_cost
        self.safety_violation_once |= self.current_safety_cost > 0
        infos = super()._record_metrics(step_reward, terminations, infos)
        infos["episode"]["safety_cost"] = to_tensor(self.safety_cost_sum.copy())
        infos["episode"]["safety_violation_once"] = to_tensor(
            self.safety_violation_once.copy()
        )
        safe_success, unsafe_success = classify_safety_success(
            self.success_once, self.safety_violation_once
        )
        infos["episode"]["safe_success_once"] = to_tensor(safe_success)
        infos["episode"]["unsafe_success_once"] = to_tensor(unsafe_success)
        return infos

    def _calc_step_reward(self, terminations):
        task_reward = super()._calc_step_reward(terminations)
        return task_reward - self.safety_cost_coef * self.current_safety_cost
