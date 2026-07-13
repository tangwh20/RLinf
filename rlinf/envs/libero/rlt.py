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

from typing import Any

import numpy as np
import torch


def get_rlt_switch_flags(
    elapsed_steps: Any,
    switch_cfg: Any | None,
) -> torch.Tensor | None:
    """Return LIBERO's full-task RLT actor gate for each environment."""
    if switch_cfg is None or not bool(switch_cfg.get("enable", False)):
        return None

    actor_start_step = int(switch_cfg.get("actor_start_step", 0))
    if actor_start_step < 0:
        raise ValueError(
            "rlt_policy_switch.actor_start_step must be non-negative, "
            f"got {actor_start_step}."
        )

    return torch.as_tensor(elapsed_steps >= actor_start_step, dtype=torch.bool)[:, None]


def get_padded_eval_reset_state_ids(
    pool: np.ndarray,
    start_idx: int,
    num_reset_states: int,
) -> tuple[np.ndarray, int]:
    """Return a fixed-size eval batch by cycling an assigned reset-state pool."""
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) == 0:
        return np.full((num_reset_states,), -1, dtype=np.int64), start_idx

    indices = (np.arange(num_reset_states, dtype=np.int64) + start_idx) % len(pool)
    return pool[indices], start_idx + num_reset_states
