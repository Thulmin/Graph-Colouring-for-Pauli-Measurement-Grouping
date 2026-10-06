"""Generate (or verify) every Hamiltonian instance listed in configs/experiments.yaml.

Each instance is generated in its own child process under the pre-registered limits
(30 minutes, 6 GiB resident memory). Failures are recorded, never replaced by synthetic data.
A log row per instance is appended to data/instances/generation_log.csv.

Usage:
    python scripts/generate_instances.py [--sets main bk exact tuning scaling]
Original code written for this project.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtp.experiments.limits import GIB, run_limited  # noqa: E402
from dtp.instances import DEFAULT_CACHE, instance_name  # noqa: E402

CHILD = ("import sys, json; from dtp.instances import get; "
         "i = get(sys.argv[1], sys.argv[2], sys.argv[3]); "
         "print(json.dumps({'name': i.name, 'n': i.n_qubits, 'm': i.m, "
         "'seconds': i.provenance['generation_seconds'], 'sha256': i.provenance['sha256']}))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["tuning", "main", "bk", "exact", "scaling"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "experiments.yaml"))
    lim = cfg["resource_limits"]
    log_path = DEFAULT_CACHE / "generation_log.csv"
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    seen = set()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["utc", "name", "status", "n_qubits", "m", "wall_seconds", "peak_rss_mb", "sha256", "note"])
        for set_name in args.sets:
            for spec in cfg["instance_sets"][set_name]:
                name = instance_name(spec["molecule"], spec["basis"], spec["encoding"])
                if name in seen:
                    continue
                seen.add(name)
                cached = (DEFAULT_CACHE / f"{name}.json").exists()
                res = run_limited([sys.executable, "-c", CHILD, spec["molecule"], spec["basis"], spec["encoding"]],
                                  timeout_s=lim["generation_seconds"], mem_bytes=int(lim["memory_gib"] * GIB),
                                  env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                       "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")})
                info = {}
                if res.status == "ok":
                    info = json.loads(res.stdout.strip().splitlines()[-1])
                note = "loaded from cache (SHA-256 verified)" if cached else ""
                if res.status in ("timeout", "memory"):
                    note = "not executed (resource limit)"
                elif res.status == "error":
                    note = res.stderr.strip().splitlines()[-1][:200] if res.stderr.strip() else "error"
                row = [datetime.now(timezone.utc).isoformat(timespec="seconds"), name, res.status,
                       info.get("n", ""), info.get("m", ""), round(res.seconds, 2),
                       round(res.peak_rss_bytes / 2 ** 20, 1), info.get("sha256", ""), note]
                w.writerow(row)
                fh.flush()
                print(*row, sep=" | ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
