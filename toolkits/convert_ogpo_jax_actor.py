# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Convert an OGPO_public Flax flow actor to RLinf PyTorch parameters."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import torch

_OGPO_PATH = (
    Path(__file__).resolve().parents[1] / "rlinf/models/embodiment/modules/ogpo.py"
)
_SPEC = importlib.util.spec_from_file_location("rlinf_ogpo_module", _OGPO_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
OGPOFlowMLPActorr = _MODULE.OGPOFlowMLPActor


def _tensor(value, *, transpose: bool = False) -> torch.Tensor:
    array = np.asarray(value)
    if transpose:
        array = array.T
    return torch.from_numpy(np.array(array, copy=True)).float()


def _copy_linear(state, prefix, flax, name):
    state[f"{prefix}.weight"] = _tensor(flax[name]["kernel"], transpose=True)
    state[f"{prefix}.bias"] = _tensor(flax[name]["bias"])


def convert_actor_params(flax_actor) -> dict[str, torch.Tensor]:
    """Return a strict PyTorch state dict for one Flax ActorVectorField."""
    actor = OGPOFlowMLPActorr(
        obs_dim=2057,
        action_dim=28,
        denoising_steps=10,
        hidden_dims=(512, 512, 512, 512),
        time_embedding_dim=32,
        two_tier_image_dim=2048,
        two_tier_proprio_dim=9,
        two_tier_fused_dim=512,
    )
    state = actor.state_dict()
    two_tier = flax_actor["two_tier_encoder"]
    mappings = (
        ("two_tier.image_projection", two_tier, "img_proj"),
        ("two_tier.proprio_projection", two_tier, "prop_proj"),
    )
    for prefix, source, name in mappings:
        _copy_linear(state, prefix, source, name)
    for torch_name, flax_name in (
        ("image_norm", "img_ln"),
        ("proprio_norm", "prop_ln"),
    ):
        state[f"two_tier.{torch_name}.weight"] = _tensor(two_tier[flax_name]["scale"])
        state[f"two_tier.{torch_name}.bias"] = _tensor(two_tier[flax_name]["bias"])
    for index in range(5):
        _copy_linear(
            state, f"velocity_net.{2 * index}", flax_actor["mlp"], f"Dense_{index}"
        )
    _copy_linear(
        state, "official_time_embedding.proj.0", flax_actor["time_embedding"], "Dense_0"
    )
    _copy_linear(
        state, "official_time_embedding.proj.2", flax_actor["time_embedding"], "Dense_1"
    )
    actor.load_state_dict(state, strict=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    params = checkpoint["agent"]["actor_network"]["params"]
    output = {
        "actor": convert_actor_params(params["modules_actor"]),
        "target_actor": convert_actor_params(params["modules_target_actor"]),
        "bc_updates": int(np.asarray(checkpoint["agent"]["actor_network"]["step"])),
        "source": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"converted {args.checkpoint} -> {args.output}")
    print(f"actor tensors: {len(output['actor'])}; BC updates: {output['bc_updates']}")


if __name__ == "__main__":
    main()
