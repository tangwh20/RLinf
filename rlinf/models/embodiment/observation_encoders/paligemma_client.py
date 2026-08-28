# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Client for a frozen PaliGemma observation-encoder service."""

from __future__ import annotations

import json
import os
import threading
import time
from multiprocessing.connection import Client

import numpy as np

from rlinf.utils.logging import get_logger

logger = get_logger()


class PaliGemmaEncoderClient:
    """Send image/proprio batches to the isolated JAX encoder process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29571,
        authkey: bytes = b"rlinf-paligemma",
        expected_output_dim: int = 2057,
    ) -> None:
        self._address = (host, int(port))
        self._authkey = authkey
        self._expected_output_dim = int(expected_output_dim)
        self._connection = None
        self._lock = threading.Lock()
        self._profile_enabled = os.environ.get("OGPO_PALIGEMMA_PROFILE", "0") == "1"
        self._profile_interval = int(
            os.environ.get("OGPO_PALIGEMMA_PROFILE_INTERVAL", "100")
        )
        self._profile_count = 0
        self._profile_sums: dict[str, float] = {}

    def _record_profile(self, **values: float) -> None:
        if not self._profile_enabled:
            return
        self._profile_count += 1
        for key, value in values.items():
            self._profile_sums[key] = self._profile_sums.get(key, 0.0) + float(value)
        if self._profile_count % self._profile_interval == 0:
            summary = {
                "component": "paligemma_client",
                "pid": os.getpid(),
                "requests": self._profile_count,
                **{
                    f"mean_{key}": total / self._profile_interval
                    for key, total in self._profile_sums.items()
                },
            }
            logger.info("PALIGEMMA_PROFILE %s", json.dumps(summary))
            self._profile_sums.clear()

    def _connect(self):
        if self._connection is None:
            self._connection = Client(self._address, authkey=self._authkey)
        return self._connection

    def health(self) -> dict:
        """Return encoder process health information."""
        with self._lock:
            connection = self._connect()
            connection.send({"op": "health"})
            response = connection.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "PaliGemma health check failed"))
        return response

    def encode(self, observations: np.ndarray, images: np.ndarray) -> np.ndarray:
        """Encode matching proprio and image batches into frozen features."""
        prepare_start_ns = time.perf_counter_ns()
        observations = np.asarray(observations, dtype=np.float32)
        images = np.asarray(images)
        if observations.ndim != 2 or images.ndim != 4:
            raise ValueError(
                "PaliGemma expects observations [B,D] and images [B,H,W,C], "
                f"got {observations.shape} and {images.shape}."
            )
        if observations.shape[0] != images.shape[0]:
            raise ValueError("PaliGemma observation/image batch sizes differ.")
        prepare_end_ns = time.perf_counter_ns()
        with self._lock:
            connection = self._connect()
            send_start_ns = time.perf_counter_ns()
            connection.send(
                {"op": "encode", "observations": observations, "images": images}
            )
            send_end_ns = time.perf_counter_ns()
            response = connection.recv()
            recv_end_ns = time.perf_counter_ns()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "PaliGemma encoding failed"))
        encoded = np.asarray(response["encoded"], dtype=np.float32)
        expected_shape = (observations.shape[0], self._expected_output_dim)
        if encoded.shape != expected_shape:
            raise RuntimeError(f"Unexpected PaliGemma output shape: {encoded.shape}")
        if not np.isfinite(encoded).all():
            raise RuntimeError("PaliGemma returned NaN or Inf.")
        server_profile = response.get("profile", {})
        self._record_profile(
            prepare_ms=(prepare_end_ns - prepare_start_ns) / 1e6,
            request_send_ms=(send_end_ns - send_start_ns) / 1e6,
            response_wait_ms=(recv_end_ns - send_end_ns) / 1e6,
            round_trip_ms=(recv_end_ns - send_start_ns) / 1e6,
            image_kib=images.nbytes / 1024,
            embedding_kib=encoded.nbytes / 1024,
            server_lock_wait_ms=server_profile.get("server_lock_wait_ms", 0.0),
            server_encode_ms=server_profile.get("server_encode_ms", 0.0),
        )
        return encoded

    def close(self) -> None:
        """Close this client's connection without stopping the shared server."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
