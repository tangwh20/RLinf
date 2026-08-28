# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Numerically compare converted RLinf and OGPO_public flow actors."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import torch


def _load_torch_actor(repo: Path, checkpoint: Path):
    module_path = repo / "rlinf/models/embodiment/modules/ogpo.py"
    spec = importlib.util.spec_from_file_location("rlinf_ogpo_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    actor = module.OGPOFlowMLPActor(
        obs_dim=2057,
        action_dim=28,
        denoising_steps=10,
        hidden_dims=(512, 512, 512, 512),
        time_embedding_dim=32,
        two_tier_image_dim=2048,
        two_tier_proprio_dim=9,
        two_tier_fused_dim=512,
    )
    actor.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["actor"], strict=True
    )
    actor.eval()
    return actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogpo-root", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    with args.jax_checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    flax_params = checkpoint["agent"]["actor_network"]["params"]["modules_actor"]
    torch_actor = _load_torch_actor(args.rlinf_root, args.torch_checkpoint)

    ng = np.random.default_rng(82723)
    observations = ng.normal(size=(7, 2057)).astype(np.float32)
    actions = ng.normal(size=(7, 28)).astype(np.float32)
    times = ng.uniform(size=(7, 1)).astype(np.float32)

    def dense(x, params):
        return x @ params["kernel"] + params["bias"]

    def layer_norm(x, params):
        normalized = (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(
            x.var(-1, keepdims=True) + 1e-6
        )
        return normalized * params["scale"] + params["bias"]

    import jax

    obs = jnp.asarray(observations)
    image = obs[:, :2048]
    image = image / (jnp.linalg.norm(image, axis=-1, keepdims=True) + 1e-6)
    tier = flax_params["two_tier_encoder"]
    image = jnn.gelu(layer_norm(dense(image, tier["img_proj"]), tier["img_ln"]))
    proprio = jnn.gelu(
        layer_norm(dense(obs[:, 2048:], tier["prop_proj"]), tier["prop_ln"])
    )
    fused = jnp.concatenate([image, proprio], axis=-1)
    half = 16
    frequencies = jnp.exp(
        jnp.arange(half, dtype=jnp.float32) * -(np.log(10000.0) / (half - 1))
    )
    angles = jnp.asarray(times)[:, 0, None] * frequencies
    time_features = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    time_params = flax_params["time_embedding"]
    time_features = dense(time_features, time_params["Dense_0"])
    time_features = jnn.gelu(time_features)
    time_features = dense(time_features, time_params["Dense_1"])
    hidden = jnp.concatenate([fused, jnp.asarray(actions), time_features], axis=-1)
    jax_fused = np.asarray(fused)
    jax_time = np.asarray(time_features)
    jax_layers = []
    for index in range(5):
        hidden = dense(hidden, flax_params["mlp"][f"Dense_{index}"])
        if index < 4:
            hidden = jnn.gelu(hidden)
        jax_layers.append(np.asarray(hidden))
    jax_output = np.asarray(hidden)
    with torch.no_grad():
        torch_fused = torch_actor.two_tier(torch.from_numpy(observations))
        torch_time = torch_actor.official_time_embedding(torch.from_numpy(times))
        torch_hidden = torch.cat(
            [torch_fused, torch.from_numpy(actions), torch_time], dim=-1
        )
        torch_layers = []
        for layer in torch_actor.velocity_net:
            torch_hidden = layer(torch_hidden)
            if isinstance(layer, torch.nn.Linear):
                torch_layers.append(torch_hidden.numpy())
        torch_output = torch_actor.predict_velocity(
            torch.from_numpy(observations),
            torch.from_numpy(actions),
            torch.from_numpy(times),
        ).numpy()
    for name, left, right in (
        ("two_tier", jax_fused, torch_fused.numpy()),
        ("time", jax_time, torch_time.numpy()),
        *[(f"mlp_{i}", jax_layers[i], torch_layers[i]) for i in range(5)],
    ):
        print(f"{name}_max_abs={np.abs(left - right).max():.9g}")
    difference = np.abs(jax_output - torch_output)
    print(f"max_abs={difference.max():.9g}")
    print(f"mean_abs={difference.mean():.9g}")
    print(f"max_rel={(difference / np.maximum(np.abs(jax_output), 1e-7)).max():.9g}")
    np.testing.assert_allclose(jax_output, torch_output, rtol=2e-5, atol=2e-5)
    print("actor velocity parity: PASS")


if __name__ == "__main__":
    main()
