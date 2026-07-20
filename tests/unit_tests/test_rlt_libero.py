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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import SimulatorRLTRoute, build_rlt_route
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.envs.libero.rlt import (
    get_padded_eval_reset_state_ids,
    get_rlt_switch_flags,
)


def _cfg(env_type: str, loss_type: str):
    return OmegaConf.create(
        {
            "env": {"train": {"env_type": env_type}},
            "algorithm": {
                "loss_type": loss_type,
                "rlt_schedule": {
                    "enable": True,
                    "warmup_post_collect_updates": 100,
                },
            },
        }
    )


def test_libero_rlt_uses_simulator_transition_route():
    cfg = _cfg("libero", "rlt_ac")

    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.use_schedule
    assert route.warmup_updates == 100


def test_libero_safety_rlt_uses_simulator_transition_route():
    cfg = _cfg("libero_safety", "rlt_ac")

    assert use_simulator_transition_replay(cfg)
    assert isinstance(build_rlt_route(cfg), SimulatorRLTRoute)


def test_non_rlt_libero_does_not_change_replay_mode():
    assert not use_simulator_transition_replay(_cfg("libero", "actor_critic"))
    assert not use_simulator_transition_replay(_cfg("libero_safety", "actor_critic"))


def test_libero_without_algorithm_config_does_not_change_replay_mode():
    cfg = OmegaConf.create({"env": {"train": {"env_type": "libero"}}})

    assert not use_simulator_transition_replay(cfg)


def test_libero_rlt_switch_flags_respect_actor_start_step():
    elapsed_steps = np.array([0, 4, 5, 8], dtype=np.int32)

    flags = get_rlt_switch_flags(
        elapsed_steps,
        {"enable": True, "actor_start_step": 5},
    )

    assert torch.equal(
        flags,
        torch.tensor([[False], [False], [True], [True]]),
    )


def test_libero_rlt_switch_can_be_disabled():
    assert get_rlt_switch_flags(np.array([10]), {"enable": False}) is None
    assert get_rlt_switch_flags(np.array([10]), None) is None


def test_libero_rlt_switch_rejects_negative_start_step():
    with pytest.raises(ValueError, match="must be non-negative"):
        get_rlt_switch_flags(
            np.array([0]),
            {"enable": True, "actor_start_step": -1},
        )


def test_padded_eval_reset_state_ids_cycle_assigned_pool():
    reset_ids, next_start_idx = get_padded_eval_reset_state_ids(
        np.array([10, 11, 12]),
        start_idx=0,
        num_reset_states=5,
    )

    assert reset_ids.tolist() == [10, 11, 12, 10, 11]
    assert next_start_idx == 5
