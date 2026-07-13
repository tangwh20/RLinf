# Copyright 2025 The RLinf Authors.
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

"""Low-overhead node-memory diagnostics for long-running jobs."""

import os
import threading
from typing import Protocol

import psutil

_INDUCTOR_WORKER_MARKER = "torch/_inductor/compile_worker"


class _Logger(Protocol):
    def info(self, message: str, *args: object) -> None: ...


def log_memory_snapshot(logger: _Logger, label: str) -> None:
    """Log node memory and PyTorch Inductor compiler-worker usage."""
    memory = psutil.virtual_memory()
    inductor_count = 0
    inductor_rss = 0
    for process in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            info = process.info
            command = " ".join(info["cmdline"] or [])
            if _INDUCTOR_WORKER_MARKER not in command:
                continue
            inductor_count += 1
            inductor_rss += info["memory_info"].rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    process_rss = psutil.Process(os.getpid()).memory_info().rss
    gib = 1024**3
    logger.info(
        "[memory-diagnostics] label=%s node_used_gib=%.1f "
        "node_available_gib=%.1f process_rss_gib=%.2f "
        "inductor_workers=%d inductor_rss_gib=%.1f",
        label,
        memory.used / gib,
        memory.available / gib,
        process_rss / gib,
        inductor_count,
        inductor_rss / gib,
    )


class MemoryDiagnosticsSampler:
    """Periodically log node-memory diagnostics until stopped."""

    def __init__(self, logger: _Logger, interval_seconds: float):
        self._logger = logger
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, label: str) -> None:
        """Start sampling and emit an initial snapshot."""
        log_memory_snapshot(self._logger, f"{label}:start")

        def sample() -> None:
            while not self._stop_event.wait(self._interval_seconds):
                log_memory_snapshot(self._logger, label)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self, label: str) -> None:
        """Stop sampling and emit a final snapshot."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1)
        log_memory_snapshot(self._logger, f"{label}:end")
