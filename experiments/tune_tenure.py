"""Tenure tuning for TabuCol on the TUNING molecules only (HF, BH3; STO-3G, JW).

The algorithm review adopted the Galinier-Hao rule t = rand(0..L-1) + floor(lambda*|CV|)
(L = 10, lambda = 0.6) "as the starting parameter rule, then tuned only on the tuning set".
This script runs that tuning with the pre-registered budget T(m), 3 seeds per setting, and
writes results/tuning/tenure_tuning.csv. Evaluation molecules are never used here.

Usage: python experiments/tune_tenure.py [--jobs 2]
Original code written for this project.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SETTINGS = [  # (label, L, lambda, fixed)
    ("GH_L10_l0.6", 10, 0.6, None),
    ("GH_L10_l1.2", 10, 1.2, None),
    ("GH_L10_l2.0", 10, 2.0, None),
    ("GH_L50_l0.6", 50, 0.6, None),
    ("GH_L100_l0.6", 100, 0.6, None),
    ("fixed10", 10, 0.6, 10),
]
MOLECULES = ["HF_sto-3g_JW", "BH3_sto-3g_JW"]
RELATIONS = ["QWC", "FC"]
SEEDS = [0, 1, 2]


def one(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from dtp import PauliSet, run_dtp
    from dtp.instances import load
    from dtp.tabucol import TabuParams
    name, rel, (label, L, lam, fixed), seed = job
    inst = load(name)
    ps = PauliSet.from_terms(inst.terms, inst.coeffs)
    params = TabuParams(tenure_L=L, tenure_lambda=lam, fixed_tenure=fixed)
    r = run_dtp(ps, rel, seed=seed, tabu_params=params, do_polish=False)
    return [name, rel, label, seed, r.m, r.k_seed, r.lb, r.k_after_reduce, r.tabu_iterations,
            round(r.timings["reduce"], 2), r.stop_reason, r.valid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    out = ROOT / "results" / "tuning" / "tenure_tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = list(itertools.product(MOLECULES, RELATIONS, SETTINGS, SEEDS))
    with open(out, "w", newline="") as fh, ProcessPoolExecutor(max_workers=args.jobs) as ex:
        w = csv.writer(fh)
        w.writerow(["instance", "relation", "setting", "seed", "m", "k_seed", "lb", "k_after_reduce",
                    "tabu_iterations", "reduce_seconds", "stop_reason", "valid"])
        for row in ex.map(one, jobs):
            w.writerow(row)
            fh.flush()
            print(*row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
