"""Phase 4 of DTP: shot-aware polishing at fixed (non-increasing) group count.

Objective: minimise S(g) = sum_c sqrt(sum_{j in group c} a_j^2), equivalently maximise
Crawford et al.'s (2021) R-hat = (sum_j |a_j| / S)^2 (Quantum 5, 385, Eq. 20).
Moves relocate one term to another *compatible, non-empty* group; they are accepted only if
S strictly decreases, so validity is preserved and the number of groups never increases.
Terms are visited in order of decreasing |a| (stable), one pass after another.

Original code written for this project.
"""
from __future__ import annotations

import time

import numpy as np

from .colouring import renumber
from .conflicts import ConflictGraph


def shot_proxy_S(col: np.ndarray, coeffs: np.ndarray) -> float:
    """S(g) = sum over groups of the Euclidean norm of the group's coefficients."""
    if col.size == 0:
        return 0.0
    s2 = np.zeros(int(col.max()) + 1)
    np.add.at(s2, col, np.asarray(coeffs, dtype=np.float64) ** 2)
    return float(np.sqrt(s2).sum())


def polish(graph: ConflictGraph, col: np.ndarray, coeffs: np.ndarray, max_passes: int = 20,
           time_limit: float = 60.0, tol: float = 1e-12, max_steps: int | None = None
           ) -> tuple[np.ndarray, dict]:
    """Relocate terms between compatible groups to reduce S. Returns (colours, stats).

    Stops after a pass without an accepted move, after ``max_passes`` passes, or when the time
    limit is reached (checked before each term visit). If ``max_steps`` is given, the time limit
    is ignored and polishing stops after that many term visits instead (deterministic replay of
    a timed run; see the G3 protocol). ``stats['steps']`` reports the term visits performed.
    """
    t0 = time.perf_counter()
    col = col.astype(np.int64, copy=True)
    m = graph.m
    if m == 0:
        return col, {"passes": 0, "moves": 0, "steps": 0, "S_before": 0.0, "S_after": 0.0,
                     "seconds": 0.0, "stopped": "empty"}
    coeffs = np.asarray(coeffs, dtype=np.float64)
    k = int(col.max()) + 1
    c2 = coeffs ** 2
    s2 = np.zeros(k)
    np.add.at(s2, col, c2)
    sizes = np.bincount(col, minlength=k)
    S_before = float(np.sqrt(s2).sum())
    order = np.argsort(-np.abs(coeffs), kind="stable")
    passes = moves = steps = 0
    stopped = "max_passes"
    out_of_budget = False
    for _ in range(max_passes):
        passes += 1
        improved = False
        for j in order.tolist():
            if max_steps is not None:
                if steps >= max_steps:
                    out_of_budget = True
                    break
            elif time.perf_counter() - t0 > time_limit:
                out_of_budget = True
                break
            steps += 1
            a = int(col[j])
            blocked = np.zeros(k, dtype=bool)
            blocked[col[graph.row(j)]] = True
            blocked[a] = True
            cand = np.flatnonzero(~blocked & (sizes > 0))
            if cand.size == 0:
                continue
            sa = s2[a]
            rest = max(sa - c2[j], 0.0) if sizes[a] > 1 else 0.0
            delta = (np.sqrt(rest) + np.sqrt(s2[cand] + c2[j])) - (np.sqrt(sa) + np.sqrt(s2[cand]))
            t = int(np.argmin(delta))
            if delta[t] < -tol:
                b = int(cand[t])
                s2[a] = rest
                s2[b] += c2[j]
                sizes[a] -= 1
                sizes[b] += 1
                col[j] = b
                moves += 1
                improved = True
        if out_of_budget:
            stopped = "steps" if max_steps is not None else "time"
            break
        if not improved:
            stopped = "converged"
            break
    col = renumber(col)
    return col, {"passes": passes, "moves": moves, "steps": steps, "S_before": S_before,
                 "S_after": shot_proxy_S(col, coeffs), "seconds": time.perf_counter() - t0,
                 "stopped": stopped}
