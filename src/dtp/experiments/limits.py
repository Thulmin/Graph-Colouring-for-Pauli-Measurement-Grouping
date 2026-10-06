"""Run a command in a child process under wall-time and resident-memory limits.

The pre-registered protocol records an instance or run as "not executed (resource limit)" when
it needs more than 30 minutes or 6 GB of RAM. This helper enforces both limits by polling the
resident set size (RSS) of the child process tree with psutil, and reports the peak RSS seen.
Original code written for this project.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass
class LimitedResult:
    """Outcome of :func:`run_limited`: status, return code, wall time, peak RSS and output."""

    status: str            # "ok", "error", "timeout" or "memory"
    returncode: int | None
    seconds: float
    peak_rss_bytes: int    # sampled peak of the process tree (polling interval below)
    stdout: str
    stderr: str


def _tree_rss(proc) -> int:
    import psutil
    try:
        total = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try:
                total += ch.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def run_limited(cmd: list[str], timeout_s: float = 1800.0, mem_bytes: int = 6 * GIB,
                env: dict | None = None, cwd: str | None = None, poll_s: float = 0.2) -> LimitedResult:
    """Run ``cmd``; kill it if it exceeds ``timeout_s`` seconds or ``mem_bytes`` of RSS."""
    import psutil
    t0 = time.perf_counter()
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=full_env, cwd=cwd)
    ps = psutil.Process(proc.pid)
    peak = 0
    status = None
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    # Read pipes in background threads so a chatty child cannot block on a full pipe.
    import threading

    def _drain(stream, sink):
        for line in iter(stream.readline, ""):
            sink.append(line)
        stream.close()

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()
    while proc.poll() is None:
        peak = max(peak, _tree_rss(ps))
        elapsed = time.perf_counter() - t0
        if elapsed > timeout_s:
            status = "timeout"
        elif peak > mem_bytes:
            status = "memory"
        if status:
            for ch in ps.children(recursive=True):
                try:
                    ch.kill()
                except psutil.Error:
                    pass
            proc.kill()
            break
        time.sleep(poll_s)
    proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    seconds = time.perf_counter() - t0
    if status is None:
        status = "ok" if proc.returncode == 0 else "error"
    return LimitedResult(status, proc.returncode, seconds, peak, "".join(out_chunks), "".join(err_chunks))
