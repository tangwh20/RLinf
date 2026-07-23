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

"""Counterfactual obstacle evaluation for LIBERO-Safety."""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np
from libero.libero.envs import OffScreenRenderEnv as SafetyOffScreenRenderEnv


def _constraint_atoms(state) -> Iterable[tuple]:
    """Yield atomic predicates from a parsed LIBERO BDDL expression."""
    if not isinstance(state, (list, tuple)) or not state:
        return
    predicate = str(state[0]).lower()
    if predicate in {"and", "or", "not"}:
        for child in state[1:]:
            yield from _constraint_atoms(child)
    else:
        yield tuple(state)


class GhostObstacleOffScreenRenderEnv(SafetyOffScreenRenderEnv):
    """Keep obstacles visible, but remove their influence on dynamics."""

    def __init__(self, *args, ghost_contact_margin=0.0, **kwargs):
        self._ghost_contact_margin = float(ghost_contact_margin)
        if self._ghost_contact_margin < 0:
            raise ValueError("ghost_contact_margin must be non-negative")
        self._ghost_ready = False
        self._ghost_robot_contact = False
        self._ghost_object_contact = False
        self._ghost_robot_pairs: list[tuple[int, int]] = []
        self._ghost_object_pairs: list[tuple[int, int]] = []
        self._ghost_hazard_indices = np.array([], dtype=np.int32)
        self._ghost_hazard_contype = np.array([], dtype=np.int32)
        self._ghost_hazard_conaffinity = np.array([], dtype=np.int32)
        super().__init__(*args, **kwargs)
        self._install_substep_hooks()
        self._configure_ghost_obstacles()

    def _install_substep_hooks(self):
        original_pre_action = self.env._pre_action
        original_post_action = self.env._post_action

        def ghost_pre_action(action, policy_step=False):
            self._record_ghost_contacts()
            return original_pre_action(action, policy_step=policy_step)

        def ghost_post_action(action):
            self._record_ghost_contacts()
            return original_post_action(action)

        self.env._pre_action = ghost_pre_action
        self.env._post_action = ghost_post_action

    def _object_geom_ids(self, object_name: str) -> list[int]:
        object_model = self.env.get_object(object_name)
        ids = []
        for geom_name in object_model.contact_geoms:
            geom_id = self.sim.model.geom_name2id(geom_name)
            if geom_id >= 0:
                ids.append(int(geom_id))
        return ids

    def _robot_geom_ids(self) -> list[int]:
        ids = []
        for geom_id in range(self.sim.model.ngeom):
            geom_name = self.sim.model.geom_id2name(geom_id)
            is_collision_geom = self.sim.model.geom_group[geom_id] == 0
            if (
                geom_name
                and is_collision_geom
                and ("robot" in geom_name or "gripper" in geom_name)
            ):
                ids.append(geom_id)
        return ids

    @staticmethod
    def _cross_pairs(left: Iterable[int], right: Iterable[int]):
        return [(int(a), int(b)) for a in left for b in right if a != b]

    def _configure_ghost_obstacles(self):
        robot_ids = self._robot_geom_ids()
        hazard_ids: set[int] = set()
        robot_pairs: set[tuple[int, int]] = set()
        object_pairs: set[tuple[int, int]] = set()

        for constraint in self.env.parsed_problem.get("constraints", []):
            for atom in _constraint_atoms(constraint):
                predicate = str(atom[0]).lower()
                if predicate == "checkrobotcontact" and len(atom) == 2:
                    obstacle_ids = self._object_geom_ids(atom[1])
                    hazard_ids.update(obstacle_ids)
                    robot_pairs.update(self._cross_pairs(robot_ids, obstacle_ids))
                elif predicate == "checkcontact" and len(atom) == 3:
                    object_ids = self._object_geom_ids(atom[1])
                    obstacle_ids = self._object_geom_ids(atom[2])
                    hazard_ids.update(obstacle_ids)
                    object_pairs.update(
                        self._cross_pairs(object_ids, obstacle_ids)
                    )

        if not hazard_ids:
            raise RuntimeError(
                "Ghost-obstacle evaluation found no CheckRobotContact or "
                "CheckContact obstacle geoms in the task constraints"
            )

        self._ghost_robot_pairs = sorted(robot_pairs)
        self._ghost_object_pairs = sorted(object_pairs)

        # Rendering remains unchanged; only contact-force participation is off.
        hazard_indices = np.fromiter(sorted(hazard_ids), dtype=np.int32)
        self._ghost_hazard_indices = hazard_indices
        self._ghost_hazard_contype = self.sim.model.geom_contype[
            hazard_indices
        ].copy()
        self._ghost_hazard_conaffinity = self.sim.model.geom_conaffinity[
            hazard_indices
        ].copy()
        self.sim.model.geom_contype[hazard_indices] = 0
        self.sim.model.geom_conaffinity[hazard_indices] = 0
        self.sim.forward()
        self._ghost_ready = True

    def _pairs_intersect(self, pairs: Iterable[tuple[int, int]]) -> bool:
        # A positive distmax distinguishes separation from exact contact when
        # the requested violation margin is zero.
        distmax = self._ghost_contact_margin + 1e-6
        fromto = np.empty(6, dtype=np.float64)
        model = getattr(self.sim.model, "_model", self.sim.model)
        data = getattr(self.sim.data, "_data", self.sim.data)
        for geom_a, geom_b in pairs:
            distance = mujoco.mj_geomDistance(
                model,
                data,
                geom_a,
                geom_b,
                distmax,
                fromto,
            )
            if distance <= self._ghost_contact_margin:
                return True
        return False

    def _record_ghost_contacts(self):
        if not self._ghost_ready:
            return
        if self._ghost_contact_margin == 0:
            self._record_exact_contacts()
            return
        self._ghost_robot_contact |= self._pairs_intersect(
            self._ghost_robot_pairs
        )
        self._ghost_object_contact |= self._pairs_intersect(
            self._ghost_object_pairs
        )

    def _record_exact_contacts(self):
        """Run collision detection without applying obstacle contact forces."""
        model = getattr(self.sim.model, "_model", self.sim.model)
        data = getattr(self.sim.data, "_data", self.sim.data)
        ids = self._ghost_hazard_indices

        self.sim.model.geom_contype[ids] = self._ghost_hazard_contype
        self.sim.model.geom_conaffinity[ids] = self._ghost_hazard_conaffinity
        mujoco.mj_collision(model, data)

        robot_pairs = set(self._ghost_robot_pairs)
        object_pairs = set(self._ghost_object_pairs)
        for contact_idx in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_idx]
            pair = (int(contact.geom1), int(contact.geom2))
            reverse_pair = (pair[1], pair[0])
            self._ghost_robot_contact |= (
                pair in robot_pairs or reverse_pair in robot_pairs
            )
            self._ghost_object_contact |= (
                pair in object_pairs or reverse_pair in object_pairs
            )

        # Clear ghost contacts before the subsequent dynamics step.
        self.sim.model.geom_contype[ids] = 0
        self.sim.model.geom_conaffinity[ids] = 0
        mujoco.mj_collision(model, data)

    def reset(self):
        obs = super().reset()
        # hard_reset rebuilds the MuJoCo model and restores collision masks.
        self._configure_ghost_obstacles()
        return obs

    def step(self, action):
        self._ghost_robot_contact = False
        self._ghost_object_contact = False
        obs, reward, done, info = super().step(action)
        robot_contact = bool(self._ghost_robot_contact)
        object_contact = bool(self._ghost_object_contact)
        info["cost"] = {
            "CheckRobotContact": int(robot_contact),
            "CheckContact": int(object_contact),
        }
        info["counterfactual_robot_contact"] = robot_contact
        info["counterfactual_object_contact"] = object_contact
        return obs, reward, done, info
