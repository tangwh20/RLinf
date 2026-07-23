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

"""Per-environment robosuite controller profiles for LIBERO variants."""

from __future__ import annotations

import contextlib
import copy

STANDARD_OSC_POSE_OVERRIDES = {
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "kp_limits": [0, 300],
}


@contextlib.contextmanager
def controller_profile(profile: str | None):
    """Temporarily override robosuite's OSC_POSE config during env creation."""
    if profile in (None, "native"):
        yield
        return
    if profile != "standard":
        raise ValueError(
            f"Unknown LIBERO controller profile {profile!r}; "
            "expected 'native' or 'standard'"
        )

    import robosuite

    original_loader = robosuite.load_controller_config

    def load_controller_config(*args, **kwargs):
        config = copy.deepcopy(original_loader(*args, **kwargs))
        controller_name = kwargs.get("default_controller")
        if controller_name is None and args:
            controller_name = args[1] if len(args) > 1 else None
        if controller_name == "OSC_POSE" or config.get("type") == "OSC_POSE":
            config.update(copy.deepcopy(STANDARD_OSC_POSE_OVERRIDES))
        return config

    robosuite.load_controller_config = load_controller_config
    try:
        yield
    finally:
        robosuite.load_controller_config = original_loader


def with_controller_profile(base_env_cls, profile: str):
    """Build an OffScreenRenderEnv subclass with a scoped controller profile."""

    class ControllerProfileEnv(base_env_cls):
        def __init__(self, *args, **kwargs):
            with controller_profile(profile):
                super().__init__(*args, **kwargs)
            controller = self.env.robots[0].controller
            self.rlinf_controller_profile = {
                "name": profile,
                "output_max": controller.output_max.tolist(),
                "output_min": controller.output_min.tolist(),
                "kp": controller.kp.tolist(),
            }

    ControllerProfileEnv.__name__ = (
        f"{profile.title()}Controller{base_env_cls.__name__}"
    )
    return ControllerProfileEnv
