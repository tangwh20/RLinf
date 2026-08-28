#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Serve OGPO_public's frozen JAX PaliGemma encoder over localhost."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np


class _ProfileAccumulator:
    """Emit low-frequency encoder timing summaries without per-call file I/O."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("OGPO_PALIGEMMA_PROFILE", "0") == "1"
        self.interval = int(os.environ.get("OGPO_PALIGEMMA_PROFILE_INTERVAL", "100"))
        self._lock = threading.Lock()
        self._count = 0
        self._sums: dict[str, float] = {}

    def add(self, **values: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._count += 1
            for key, value in values.items():
                self._sums[key] = self._sums.get(key, 0.0) + float(value)
            if self._count % self.interval == 0:
                summary = {
                    "component": "paligemma_server",
                    "requests": self._count,
                    **{
                        f"mean_{key}": total / self.interval
                        for key, total in self._sums.items()
                    },
                }
                print("PALIGEMMA_PROFILE " + json.dumps(summary), flush=True)
                self._sums.clear()


_PROFILE = _ProfileAccumulator()


def parse_args() -> argparse.Namespace:
    """Parse server options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogpo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--env-name", default="square-mh-image")
    parser.add_argument("--prompt")
    parser.add_argument("--output-dim", default=2057, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=29571, type=int)
    return parser.parse_args()


def serve_connection(
    connection, encode_fn, encode_lock: threading.Lock, output_dim: int
) -> None:
    """Serve requests from one RLinf environment process."""
    try:
        while True:
            request = connection.recv()
            operation = request.get("op")
            if operation == "health":
                connection.send({"ok": True, "output_dim": output_dim})
                continue
            if operation != "encode":
                connection.send({"ok": False, "error": f"Unknown op: {operation}"})
                continue
            try:
                received_ns = time.perf_counter_ns()
                lock_start_ns = received_ns
                with encode_lock:
                    encode_start_ns = time.perf_counter_ns()
                    encoded = np.asarray(
                        encode_fn(request["observations"], request["images"]),
                        dtype=np.float32,
                    )
                    encode_end_ns = time.perf_counter_ns()
                response = {"ok": True, "encoded": encoded}
                if _PROFILE.enabled:
                    response["profile"] = {
                        "server_lock_wait_ms": (encode_start_ns - lock_start_ns) / 1e6,
                        "server_encode_ms": (encode_end_ns - encode_start_ns) / 1e6,
                    }
                send_start_ns = time.perf_counter_ns()
                connection.send(response)
                send_end_ns = time.perf_counter_ns()
                _PROFILE.add(
                    lock_wait_ms=(encode_start_ns - lock_start_ns) / 1e6,
                    encode_ms=(encode_end_ns - encode_start_ns) / 1e6,
                    response_send_ms=(send_end_ns - send_start_ns) / 1e6,
                    request_to_response_ms=(send_end_ns - received_ns) / 1e6,
                    image_kib=request["images"].nbytes / 1024,
                    embedding_kib=encoded.nbytes / 1024,
                )
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
    except (EOFError, ConnectionResetError):
        pass
    finally:
        connection.close()


def main() -> None:
    """Load PaliGemma once and serve concurrent clients serially on the GPU."""
    args = parse_args()
    module_path = (
        args.ogpo_root.resolve() / "ogpo" / "networks" / "encoders" / "paligemma.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("rlinf_frozen_paligemma", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PaliGemma encoder from {module_path}")
    paligemma = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = paligemma
    spec.loader.exec_module(paligemma)

    if args.prompt:
        paligemma.ROBOMIMIC_PROMPTS[args.env_name] = args.prompt

    encode_fn = paligemma.load_paligemma_encoder(
        checkpoint_path=str(args.checkpoint.resolve()),
        env_name=args.env_name,
        img_proj_dim=0,
        state_proj_dim=0,
        encoder_device=0,
        tokenizer_path=str(args.tokenizer.resolve()),
    )
    encode_lock = threading.Lock()
    listener = Listener(
        (args.host, args.port), family="AF_INET", authkey=b"rlinf-paligemma"
    )
    print(f"PaliGemma encoder ready on {args.host}:{args.port}", flush=True)
    try:
        while True:
            connection = listener.accept()
            threading.Thread(
                target=serve_connection,
                args=(connection, encode_fn, encode_lock, args.output_dim),
                daemon=True,
            ).start()
    finally:
        listener.close()


if __name__ == "__main__":
    main()
