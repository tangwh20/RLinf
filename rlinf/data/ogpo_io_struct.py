# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OGPO-only rollout payloads for primitive-step sequence reconstruction."""

from dataclasses import dataclass, field
from typing import Any

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult, Trajectory


def _extract_chunk_successes(
    infos_list: list[dict[str, Any]], batch_size: int
) -> torch.Tensor:
    """Extract per-step successes, including terminal auto-reset information.

    Vector environments move the terminal step's info under ``final_info`` when
    they auto-reset. Preserve top-level successes for active environments and
    restore terminal successes for the environments selected by
    ``_final_info``.
    """

    def _as_batch_tensor(value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(batch_size)
        return tensor.reshape(batch_size)

    success_steps = []
    for info in infos_list:
        success = torch.zeros(batch_size, dtype=torch.float32)
        if isinstance(info, dict) and info.get("success") is not None:
            success = _as_batch_tensor(info["success"])

        final_info = info.get("final_info") if isinstance(info, dict) else None
        final_success = (
            final_info.get("success") if isinstance(final_info, dict) else None
        )
        if final_success is not None:
            terminal_success = _as_batch_tensor(final_success)
            final_mask = info.get("_final_info")
            if final_mask is None:
                success = terminal_success
            else:
                mask = torch.as_tensor(final_mask, dtype=torch.bool).reshape(batch_size)
                success = torch.where(mask, terminal_success, success)
        success_steps.append(success)

    return torch.stack(success_steps, dim=1)


@dataclass
class OGPOTrajectory(Trajectory):
    """Trajectory carrying primitive transitions required by OGPO replay."""

    source_rank: int = -1
    valid: torch.Tensor = None
    primitive_curr_states: torch.Tensor = None
    primitive_next_states: torch.Tensor = None
    primitive_actions: torch.Tensor = None
    primitive_rewards: torch.Tensor = None
    primitive_successes: torch.Tensor = None
    primitive_terminations: torch.Tensor = None
    primitive_truncations: torch.Tensor = None


@dataclass(kw_only=True)
class OGPOEmbodiedRolloutResult(EmbodiedRolloutResult):
    """Collect OGPO primitive transitions without extending shared IO types."""

    source_rank: int = -1
    primitive_curr_states: list[torch.Tensor] = field(default_factory=list)
    primitive_next_states: list[torch.Tensor] = field(default_factory=list)
    primitive_actions: list[torch.Tensor] = field(default_factory=list)
    primitive_rewards: list[torch.Tensor] = field(default_factory=list)
    primitive_successes: list[torch.Tensor] = field(default_factory=list)
    primitive_terminations: list[torch.Tensor] = field(default_factory=list)
    primitive_truncations: list[torch.Tensor] = field(default_factory=list)

    _PRIMITIVE_FIELDS = (
        "primitive_curr_states",
        "primitive_next_states",
        "primitive_actions",
        "primitive_rewards",
        "primitive_successes",
        "primitive_terminations",
        "primitive_truncations",
    )

    def append_primitive_transitions(
        self,
        *,
        curr_states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        successes: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ) -> None:
        """Append one action chunk as primitive-step tensors."""
        values = {
            "primitive_curr_states": curr_states,
            "primitive_next_states": next_states,
            "primitive_actions": actions,
            "primitive_rewards": rewards,
            "primitive_successes": successes,
            "primitive_terminations": terminations,
            "primitive_truncations": truncations,
        }
        for field_name, value in values.items():
            getattr(self, field_name).append(
                torch.as_tensor(value).detach().cpu().contiguous()
            )

    def clear(self) -> None:
        super().clear()
        for field_name in self._PRIMITIVE_FIELDS:
            getattr(self, field_name).clear()

    def to_trajectory(self) -> OGPOTrajectory:
        base_trajectory = super().to_trajectory()
        trajectory = OGPOTrajectory(
            **{
                field_name: getattr(base_trajectory, field_name)
                for field_name in Trajectory.__dataclass_fields__
            },
            source_rank=self.source_rank,
        )
        for field_name in self._PRIMITIVE_FIELDS:
            values = getattr(self, field_name)
            if values:
                setattr(
                    trajectory,
                    field_name,
                    torch.stack(values, dim=0).cpu().contiguous(),
                )
        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[OGPOTrajectory]:
        """Split without degrading OGPO fields to dynamic base-class attributes."""
        base_splits = super().to_splited_trajectories(split_size)
        return [
            OGPOTrajectory(
                **{
                    field_name: getattr(trajectory, field_name, None)
                    for field_name in OGPOTrajectory.__dataclass_fields__
                }
            )
            for trajectory in base_splits
        ]
