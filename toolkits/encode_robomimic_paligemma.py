#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
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

"""Persist frozen OGPO PaliGemma features for a Robomimic dataset.

This tool deliberately imports the encoder from OGPO_public so the cached
features use exactly the same preprocessing, prompt, and frozen parameters as
the reference implementation. It writes to ``<output>.partial`` and atomically
renames the file only after validation succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ogpo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--env-name", default="square-mh-image")
    parser.add_argument("--image-key", default="agentview_image")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_demos(data: h5py.Group) -> list[str]:
    """Return Robomimic demo names in numeric order."""
    return sorted(data.keys(), key=lambda name: int(name.removeprefix("demo_")))


def concatenate_proprio(group: h5py.Group, side: str) -> np.ndarray:
    """Read OGPO_public's nine-dimensional Square proprioception."""
    return np.concatenate(
        [np.asarray(group[f"{side}/{key}"], dtype=np.float32) for key in PROPRIO_KEYS],
        axis=-1,
    )


def create_output(
    output: h5py.File,
    *,
    size: int,
    observation_dim: int,
    action_dim: int,
    compression: str | None,
) -> None:
    """Create the flat encoded dataset schema consumed by RLinf."""
    chunk_rows = min(256, size)
    output.create_dataset(
        "observations",
        (size, observation_dim),
        dtype=np.float32,
        chunks=(chunk_rows, observation_dim),
        compression=compression,
    )
    output.create_dataset(
        "next_observations",
        (size, observation_dim),
        dtype=np.float32,
        chunks=(chunk_rows, observation_dim),
        compression=compression,
    )
    output.create_dataset(
        "actions",
        (size, action_dim),
        dtype=np.float32,
        chunks=(chunk_rows, action_dim),
        compression=compression,
    )
    for key in ("rewards", "terminals", "masks", "dones_float"):
        output.create_dataset(
            key,
            (size,),
            dtype=np.float32,
            chunks=(min(4096, size),),
            compression=compression,
        )


def validate_output(path: Path, expected_size: int, expected_dim: int) -> None:
    """Validate shapes, completion metadata, and finite encoded values."""
    with h5py.File(path, "r") as dataset:
        if int(dataset.attrs["complete"]) != 1:
            raise RuntimeError("Encoded dataset is not marked complete")
        if dataset["observations"].shape != (expected_size, expected_dim):
            raise RuntimeError("Unexpected observations shape")
        if dataset["next_observations"].shape != (expected_size, expected_dim):
            raise RuntimeError("Unexpected next_observations shape")
        for start in range(0, expected_size, 4096):
            end = min(start + 4096, expected_size)
            if not np.isfinite(dataset["observations"][start:end]).all():
                raise RuntimeError(f"Non-finite observation in rows [{start}, {end})")
            if not np.isfinite(dataset["next_observations"][start:end]).all():
                raise RuntimeError(
                    f"Non-finite next observation in rows [{start}, {end})"
                )


def main() -> None:
    """Encode the dataset and atomically publish the validated cache."""
    args = parse_args()
    for path in (args.input, args.ogpo_root, args.checkpoint, args.tokenizer):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_path = args.output.resolve()
    partial_path = Path(f"{output_path}.partial")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if partial_path.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial_path}. Inspect or remove it before retrying."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.ogpo_root.resolve()))
    from ogpo.networks.encoders.paligemma import load_paligemma_encoder

    encode_fn = load_paligemma_encoder(
        checkpoint_path=str(args.checkpoint.resolve()),
        env_name=args.env_name,
        img_proj_dim=0,
        state_proj_dim=0,
        encoder_device=0,
        tokenizer_path=str(args.tokenizer.resolve()),
    )
    compression = None if args.compression == "none" else args.compression

    with h5py.File(args.input, "r") as source:
        demos = sorted_demos(source["data"])
        sizes = [int(source[f"data/{demo}/actions"].shape[0]) for demo in demos]
        total_size = sum(sizes)
        first = source[f"data/{demos[0]}"]
        sample_state = concatenate_proprio(first, "obs")[:1]
        sample_image = np.asarray(first[f"obs/{args.image_key}"][:1])
        sample_encoded = np.asarray(
            encode_fn(sample_state, sample_image), dtype=np.float32
        )
        observation_dim = int(sample_encoded.shape[-1])
        action_dim = int(first["actions"].shape[-1])

        expected_dim = 2048 + sample_state.shape[-1]
        if observation_dim != expected_dim:
            raise RuntimeError(
                f"Official single-camera Square encoding should be {expected_dim}D, "
                f"got {observation_dim}D"
            )

        with h5py.File(partial_path, "w") as output:
            create_output(
                output,
                size=total_size,
                observation_dim=observation_dim,
                action_dim=action_dim,
                compression=compression,
            )
            output.attrs["complete"] = 0
            output.attrs["metadata_json"] = json.dumps(
                {
                    "format_version": 1,
                    "source_path": str(args.input.resolve()),
                    "env_name": args.env_name,
                    "encoder": "ogpo_public_paligemma",
                    "checkpoint_path": str(args.checkpoint.resolve()),
                    "tokenizer_path": str(args.tokenizer.resolve()),
                    "image_keys": [args.image_key],
                    "proprio_keys": list(PROPRIO_KEYS),
                    "image_feature_dim": 2048,
                    "proprio_dim": int(sample_state.shape[-1]),
                    "observation_dim": observation_dim,
                    "num_demos": len(demos),
                    "num_transitions": total_size,
                },
                sort_keys=True,
            )

            cursor = 0
            parity_rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            parity_demo_ids = {0, len(demos) // 2, len(demos) - 1}
            for demo_id, (demo, demo_size) in enumerate(zip(demos, sizes, strict=True)):
                group = source[f"data/{demo}"]
                obs_state = concatenate_proprio(group, "obs")
                next_state = concatenate_proprio(group, "next_obs")
                obs_images = group[f"obs/{args.image_key}"]
                next_images = group[f"next_obs/{args.image_key}"]

                for local_start in range(0, demo_size, args.batch_size):
                    local_end = min(local_start + args.batch_size, demo_size)
                    global_start = cursor + local_start
                    global_end = cursor + local_end
                    encoded_obs = np.asarray(
                        encode_fn(
                            obs_state[local_start:local_end],
                            obs_images[local_start:local_end],
                        ),
                        dtype=np.float32,
                    )
                    output["observations"][global_start:global_end] = encoded_obs
                    output["next_observations"][global_start:global_end] = encode_fn(
                        next_state[local_start:local_end],
                        next_images[local_start:local_end],
                    )

                actions = np.asarray(group["actions"], dtype=np.float32)
                terminals = np.asarray(group["dones"], dtype=np.float32)
                rewards = np.asarray(group["rewards"], dtype=np.float32) - 1.0
                dones_float = terminals.copy()
                if demo_size:
                    dones_float[-1] = 1.0
                row = slice(cursor, cursor + demo_size)
                output["actions"][row] = np.clip(actions, -1 + 1e-5, 1 - 1e-5)
                output["rewards"][row] = rewards
                output["terminals"][row] = terminals
                output["masks"][row] = 1.0 - terminals
                output["dones_float"][row] = dones_float

                if demo_id in parity_demo_ids and demo_size:
                    parity_rows.append(
                        (
                            cursor,
                            obs_state[:1].copy(),
                            np.asarray(obs_images[:1]).copy(),
                            np.asarray(output["observations"][cursor]).copy(),
                        )
                    )
                cursor += demo_size
                output.attrs["completed_demos"] = demo_id + 1
                output.attrs["completed_transitions"] = cursor
                output.flush()
                print(
                    f"Encoded {demo_id + 1}/{len(demos)} demos ({cursor}/{total_size} transitions)"
                )

            singleton_cosines = []
            singleton_max_abs_errors = []
            for row_id, state, image, batch_expected in parity_rows:
                persisted = np.asarray(output["observations"][row_id])
                np.testing.assert_array_equal(persisted, batch_expected)

                singleton = np.asarray(encode_fn(state, image), dtype=np.float32)[0]
                cosine = float(
                    np.dot(persisted, singleton)
                    / (np.linalg.norm(persisted) * np.linalg.norm(singleton))
                )
                max_abs_error = float(np.max(np.abs(persisted - singleton)))
                if cosine < 0.9999 or max_abs_error > 0.02:
                    raise RuntimeError(
                        "Batch/singleton PaliGemma mismatch exceeds tolerance: "
                        f"row={row_id}, cosine={cosine}, max_abs_error={max_abs_error}"
                    )
                singleton_cosines.append(cosine)
                singleton_max_abs_errors.append(max_abs_error)
            output.attrs["parity_samples"] = len(parity_rows)
            output.attrs["singleton_cosine_min"] = min(singleton_cosines)
            output.attrs["singleton_max_abs_error"] = max(singleton_max_abs_errors)
            output.attrs["complete"] = 1
            output.flush()

    validate_output(partial_path, total_size, observation_dim)
    if output_path.exists():
        output_path.unlink()
    os.replace(partial_path, output_path)
    print(
        f"Wrote validated encoded dataset: {output_path} "
        f"({total_size} transitions, observation_dim={observation_dim})"
    )


if __name__ == "__main__":
    main()
