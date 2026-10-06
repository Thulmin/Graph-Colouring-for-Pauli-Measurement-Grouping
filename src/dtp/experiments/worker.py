"""Run ONE experiment job and print its result as a single JSON line on stdout.

Usage (normally called by experiments/run_experiments.py inside a resource-limited child process):
    python -m dtp.experiments.worker '<job json>'

A job is a dict with at least: job_id, set, instance, relation, method, seed. Method-specific keys:
    dtp:              ablation ("none" | "A1" | "A2" | "A3" | "A5" | "A6"), ig_budget, iteration_budget,
                      polish_steps, trace_memory, budget (optional override; default T(m))
    iterated_greedy:  budget (optional; default T(m))
    gcol_tabucol / gcol_partialcol: it_limit
    cpsat:            time_limit, workers
Every result row carries the instance SHA-256, seed, software versions and hardware description
(requirement R16), the independent checker verdict, k, R-hat, timings and peak memory.
Original code written for this project.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time


def _versions() -> dict:
    out = {"python": platform.python_version()}
    import importlib
    for mod in ("numpy", "dtp", "qiskit", "pennylane", "rustworkx", "networkx", "gcol", "ortools",
                "pyscf", "openfermion"):
        if mod in sys.modules:
            m = sys.modules[mod]
            out[mod] = getattr(m, "__version__", "?")
    try:
        from importlib.metadata import version
        if "gcol" in sys.modules:
            out["gcol"] = version("gcol")
    except Exception:  # pragma: no cover - metadata lookup is best-effort
        pass
    return out


def _hardware() -> dict:
    cpu = platform.processor() or "?"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem = None
    try:
        with open("/proc/meminfo") as fh:
            mem = int(fh.readline().split()[1]) * 1024
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "mem_total_bytes": mem,
            "platform": platform.platform()}


def run_job(job: dict) -> dict:
    """Execute one job (see module docstring) in this process and return its result row."""
    import numpy as np

    from dtp import ConflictGraph, PauliSet, check_grouping, groups_from_colours, rhat, run_dtp
    from dtp import baselines as B
    from dtp.colouring import dsatur, greedy_clique, num_colours
    from dtp.dtp import default_budget
    from dtp.instances import load
    from dtp.polish import shot_proxy_S
    from dtp.tabucol import TabuParams

    rss_imports_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    inst = load(job["instance"])
    pset = PauliSet.from_terms(inst.terms, inst.coeffs)
    rel = job["relation"]
    method = job["method"]
    seed = int(job.get("seed", 0))
    row: dict = {**job, "m": pset.m, "n": pset.n, "instance_sha256": inst.provenance["sha256"],
                 "T_m": default_budget(pset.m)}
    w0, c0 = time.perf_counter(), time.process_time()
    extra: dict = {}
    if method == "dtp":
        abl = job.get("ablation", "none")
        kw = dict(seed=seed, budget=job.get("budget"), trace_memory=bool(job.get("trace_memory", False)),
                  iteration_budget=job.get("iteration_budget"), ig_budget=job.get("ig_budget"),
                  polish_steps=job.get("polish_steps"))
        if abl == "A1":
            kw["seed_method"] = "random"
        elif abl == "A2":
            kw["clique_stop"] = False
        elif abl == "A3":
            kw["gamma"] = "full"
        elif abl == "A5":
            kw["tabu_params"] = TabuParams(fixed_tenure=10)
        elif abl == "A6":
            kw["phase3"] = "tabu"
        elif abl != "none":
            raise ValueError(f"unknown ablation {abl!r}")
        res = run_dtp(pset, rel, **kw)
        col = np.asarray(res.colours, dtype=np.int64)
        extra = res.summary()
        for key in ("relation", "m", "n", "k", "valid", "check_message", "rhat", "S"):
            extra.pop(key, None)
    else:
        graph = None
        if method not in ("qiskit_lf",) and not method.startswith("pennylane_"):
            graph = ConflictGraph(pset, rel)
        if method == "qiskit_lf":
            out = B.qiskit_lf(pset, rel)
        elif method.startswith("pennylane_"):
            out = B.pennylane_groups(pset, rel, method.split("_", 1)[1])
        elif method == "random_greedy":
            out = B.random_greedy(graph, seed)
        elif method == "smallest_last":
            out = B.smallest_last(graph)
        elif method == "sorted_insertion":
            out = B.sorted_insertion(graph, pset.coeffs)
        elif method == "iterated_greedy":
            t = time.perf_counter()
            col0 = dsatur(graph)
            lb, _ = greedy_clique(graph)
            setup = time.perf_counter() - t
            budget = job.get("budget") or default_budget(pset.m)
            out = B.iterated_greedy(graph, col0, budget, seed, lb=lb)
            out.extra.update({"k_seed": num_colours(col0), "lb": lb, "setup_seconds": setup, "budget_s": budget})
        elif method in ("gcol_tabucol", "gcol_partialcol"):
            out = B.gcol_colouring(graph, 2 if method == "gcol_tabucol" else 3, int(job["it_limit"]), seed)
        elif method == "cpsat":
            ub = dsatur(graph)
            _, clique = greedy_clique(graph)
            out = B.cpsat_exact(graph, ub, clique, time_limit=float(job.get("time_limit", 600)),
                                workers=int(job.get("workers", 2)), seed=seed)
        else:
            raise ValueError(f"unknown method {method!r}")
        col = np.asarray(out.colours, dtype=np.int64)
        extra = {"core_seconds": out.seconds, "core_cpu_seconds": out.cpu_seconds, **out.extra}
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    groups = groups_from_colours(col)
    ok, msg = check_grouping(pset.strings, groups, rel)   # independent verdict for every method
    row.update({
        "k": len(groups), "valid": ok, "check_message": msg,
        "rhat": rhat(groups, pset.coeffs), "S": shot_proxy_S(col, pset.coeffs) if col.size else 0.0,
        "wall_seconds": wall, "cpu_seconds": cpu,
        "rss_after_imports_kb": rss_imports_kb,
        "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "colours_sha256": hashlib.sha256(np.ascontiguousarray(col, dtype=np.int64).tobytes()).hexdigest(),
        "colours": col.tolist(),
        "versions": _versions(), "hardware": _hardware(),
        "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **{f"x_{k}": v for k, v in extra.items()},
    })
    return row


def main(argv: list[str]) -> int:
    """Command-line entry point: run the job given as JSON in argv[1] and print one JSON line."""
    job = json.loads(argv[1])
    row = run_job(job)
    sys.stdout.write(json.dumps(row, default=float) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
