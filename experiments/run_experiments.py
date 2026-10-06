"""Run the pre-registered experiments (resumable; results appended to results/raw/<phase>.jsonl).

Phases (run in this order by default):
  main      evaluation set, QWC and FC: deployed options, classical baselines, DTP, Iterated Greedy
  g3        determinism gate: deterministic replays of each seed-0 DTP run (+1 timed repeat, informational)
  bk        Bravyi-Kitaev robustness set (QWC)
  exact     exact-optimum set (QWC and FC): CP-SAT, DTP and baselines
  ablation  A1 (random seed), A2 (no clique stop), A3 (full gamma), A5 (fixed tenure), A6 (TabuCol-only
            Phase 3, as first pre-registered) on the main QWC set; A4 (no polishing) is read from the
            main runs (Phase 3 output before polishing); "IG stage only" equals the Iterated Greedy
            baseline (same seed stream, start and stop rule). Change C16 (docs/DEVIATIONS.md) adds
            same-phase DTP control runs (ablation "none", same seeds), so that each ablation is
            compared with DTP runs made under the same machine load
  scaling   6-31G instances (QWC): DTP and cheap baselines
  gcol      GCol DSATUR + TabuCol / PartialCol on the evaluation set (3 seeds, fixed iteration limit, 600 s)

Every job runs in its own child process under wall-time and resident-memory limits. Jobs that
exceed the limits are recorded as "not executed (resource limit)" and never replaced.

Usage: python experiments/run_experiments.py [--phases main g3 ...] [--jobs 2] [--dry-run]
Original code written for this project.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtp.experiments.limits import GIB, run_limited  # noqa: E402
from dtp.instances import instance_name, load  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs" / "experiments.yaml"))
RAW = ROOT / "results" / "raw"
DEPLOYED = ["qiskit_lf", "pennylane_lf", "pennylane_dsatur", "pennylane_rlf", "pennylane_gis"]
HEAVY_METHODS = set(DEPLOYED) | {"gcol_tabucol", "gcol_partialcol", "cpsat"}
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "RAYON_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
HEAVY_M = 2500        # heavy methods on larger instances run alone with the full memory limit


def seeds_for(m: int) -> list[int]:
    s = CFG["seeds"]
    return list(s["small"]) if m <= s["small_m_limit"] else list(s["large"])


def inst_list(set_name: str) -> list[str]:
    return [instance_name(d["molecule"], d["basis"], d["encoding"]) for d in CFG["instance_sets"][set_name]]


_M_CACHE: dict[str, int] = {}


def m_of(name: str) -> int:
    if name not in _M_CACHE:
        _M_CACHE[name] = load(name, verify=False).m
    return _M_CACHE[name]


def job(phase, inst, rel, method, seed=0, **kw) -> dict:
    tag = kw.get("ablation", "none")
    variant = kw.pop("variant", "")
    jid = f"{phase}|{inst}|{rel}|{method}|{tag}|seed{seed}" + (f"|{variant}" if variant else "")
    return {"job_id": jid, "set": phase, "instance": inst, "relation": rel, "method": method,
            "seed": seed, "variant": variant, **kw}


def read_rows(phase: str) -> list[dict]:
    p = RAW / f"{phase}.jsonl"
    if not p.exists():
        return []
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plan(phase: str) -> list[dict]:
    jobs: list[dict] = []
    it_limit = int(CFG["gcol_it_limit"])
    if phase == "main":
        for inst in inst_list("main"):
            seeds = seeds_for(m_of(inst))
            for rel in ("QWC", "FC"):
                for meth in DEPLOYED + ["smallest_last", "sorted_insertion"]:
                    jobs.append(job(phase, inst, rel, meth))
                for s in seeds:
                    jobs.append(job(phase, inst, rel, "dtp", s))
                    jobs.append(job(phase, inst, rel, "iterated_greedy", s))
                    jobs.append(job(phase, inst, rel, "random_greedy", s))
    elif phase == "g3":
        base = {(r["instance"], r["relation"]): r for r in read_rows("main")
                if r.get("method") == "dtp" and r.get("seed") == 0 and r.get("status", "ok") == "ok"}
        for (inst, rel), r in sorted(base.items()):
            for rep in (1, 2, 3):
                jobs.append(job(phase, inst, rel, "dtp", 0, variant=f"replay{rep}",
                                ig_budget=int(r["x_ig_steps"]), iteration_budget=int(r["x_tabu_iterations"]),
                                polish_steps=int(r["x_polish_steps"]), trace_memory=True))
            jobs.append(job(phase, inst, rel, "dtp", 0, variant="timed_repeat"))
    elif phase == "bk":
        for inst in inst_list("bk"):
            seeds = seeds_for(m_of(inst))
            for meth in DEPLOYED + ["sorted_insertion"]:
                jobs.append(job(phase, inst, "QWC", meth))
            for s in seeds:
                jobs.append(job(phase, inst, "QWC", "dtp", s))
                jobs.append(job(phase, inst, "QWC", "iterated_greedy", s))
    elif phase == "exact":
        for inst in inst_list("exact"):
            seeds = seeds_for(m_of(inst))
            for rel in ("QWC", "FC"):
                jobs.append(job(phase, inst, rel, "cpsat", 0, time_limit=CFG["budgets"]["cpsat_seconds"], workers=2))
                for meth in DEPLOYED + ["sorted_insertion"]:
                    jobs.append(job(phase, inst, rel, meth))
                for s in seeds:
                    jobs.append(job(phase, inst, rel, "dtp", s))
    elif phase == "ablation":
        main_rows = [r for r in read_rows("main") if r.get("method") == "dtp" and r.get("relation") == "QWC"]
        reached = {r["instance"] for r in main_rows if r.get("x_stop_reason") == "lb_reached"}
        for inst in inst_list("main"):
            seeds = seeds_for(m_of(inst))
            for abl in ("A1", "A3", "A5", "A6"):
                for s in seeds:
                    jobs.append(job(phase, inst, "QWC", "dtp", s, ablation=abl))
            for s in seeds:  # C16: same-phase DTP control runs (no ablation)
                jobs.append(job(phase, inst, "QWC", "dtp", s, ablation="none"))
            if inst in reached:  # A2 only changes runs in which the clique stop fired
                for s in seeds:
                    jobs.append(job(phase, inst, "QWC", "dtp", s, ablation="A2"))
    elif phase == "gcol":
        for inst in inst_list("main"):
            for rel in ("QWC", "FC"):
                for meth in ("gcol_tabucol", "gcol_partialcol"):
                    for s in CFG["gcol_seeds"]:
                        jobs.append(job(phase, inst, rel, meth, s, it_limit=it_limit))
    elif phase == "scaling":
        for inst in inst_list("scaling"):
            for s in CFG["seeds"]["large"]:
                jobs.append(job(phase, inst, "QWC", "dtp", s))
            jobs.append(job(phase, inst, "QWC", "sorted_insertion"))
    else:
        raise ValueError(phase)
    return jobs


def is_heavy(j: dict) -> bool:
    if j["set"] == "scaling" or j["method"] == "cpsat":
        return True
    return j["method"] in HEAVY_METHODS and m_of(j["instance"]) > HEAVY_M


_lock = threading.Lock()


def execute(j: dict, mem_gib: float, out_path: Path) -> dict:
    lim = CFG["gcol_timeout_s"] if j["method"].startswith("gcol") else CFG["budgets"]["run_timeout_s"]
    res = run_limited([sys.executable, "-m", "dtp.experiments.worker", json.dumps(j)],
                      timeout_s=lim, mem_bytes=int(mem_gib * GIB), env=ENV, cwd=str(ROOT))
    if res.status == "ok":
        row = json.loads(res.stdout.strip().splitlines()[-1])
        row["status"] = "ok"
    else:
        row = dict(j)
        row["status"] = res.status
        row["note"] = ("not executed (resource limit)" if res.status in ("timeout", "memory")
                       else (res.stderr.strip().splitlines() or ["error"])[-1][:300])
    row["child_peak_rss_bytes"] = res.peak_rss_bytes
    row["child_wall_seconds"] = res.seconds
    row["memory_limit_gib"] = mem_gib
    return row


def append(out_path: Path, row: dict) -> None:
    with _lock:
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")


def run_phase(phase: str, n_jobs: int, dry: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / f"{phase}.jsonl"
    done = {r["job_id"] for r in read_rows(phase) if r.get("status") in ("ok", "timeout", "memory", "error")}
    todo = [j for j in plan(phase) if j["job_id"] not in done]
    light = [j for j in todo if not is_heavy(j)]
    heavy = [j for j in todo if is_heavy(j)]
    print(f"[{phase}] planned {len(todo) + len(done)} jobs, {len(done)} done, "
          f"{len(light)} light + {len(heavy)} heavy to run", flush=True)
    if dry:
        for j in todo:
            print(j["job_id"])
        return
    mem_total = CFG["resource_limits"]["memory_gib"]
    retry: list[dict] = []

    def run_light(j):
        row = execute(j, mem_total / max(1, n_jobs) if n_jobs > 1 else mem_total, out_path)
        if row["status"] == "memory" and n_jobs > 1:
            retry.append(j)            # re-run alone with the full limit
            print(f"  retry alone (memory): {j['job_id']}", flush=True)
            return
        append(out_path, row)
        print(f"  {row['status']:7s} k={row.get('k', '-')} {j['job_id']} ({row['child_wall_seconds']:.1f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        list(ex.map(run_light, light))
    failed_groups: set = set()
    for j in heavy + retry:
        grp = (j["instance"], j["relation"], j["method"], j.get("ablation", "none"), j.get("variant", ""))
        if grp in failed_groups:
            row = dict(j)
            row.update({"status": "memory", "note": "not executed (resource limit): an earlier seed of "
                        "the same method and instance exceeded the limit", "child_peak_rss_bytes": 0,
                        "child_wall_seconds": 0.0, "memory_limit_gib": mem_total, "skipped": True})
        else:
            row = execute(j, mem_total, out_path)
            if row["status"] in ("timeout", "memory"):
                failed_groups.add(grp)
        append(out_path, row)
        print(f"  {row['status']:7s} k={row.get('k', '-')} {j['job_id']} ({row['child_wall_seconds']:.1f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["main", "g3", "bk", "exact", "ablation", "scaling", "gcol"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for ph in args.phases:
        run_phase(ph, args.jobs, args.dry_run)
    print(f"finished in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
