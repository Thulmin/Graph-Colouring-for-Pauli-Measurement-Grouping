"""DSATUR-Tabu-Polish (DTP): the proposed Pauli measurement-grouping pipeline.

Phases (design dossier, sections 9-10):
  0 ENCODE   validate, remove identity, merge duplicates, bit-encode   (pauli.PauliSet)
  1 SEED     DSATUR colouring of the conflict graph                    (colouring.dsatur)
  2 BOUND    greedy clique lower bound                                 (colouring.greedy_clique)
  3 REDUCE   alternation of Iterated Greedy re-colouring and class      (iterated_greedy.ig_step,
             removal + TabuCol, until k = LB or the budget ends          tabucol.reduce_one)
  4 POLISH   shot-aware relocation at non-increasing k                 (polish.polish)
  5 VERIFY   independent letter-level checker                          (checker.check_grouping)

Phase 3 (amended before any evaluation run, see docs/DEVIATIONS.md): Iterated Greedy steps are
applied while they keep finding fewer colours; after ``ig_stall`` consecutive steps without
improvement one class-removal + TabuCol round (at most ``tabu_cap`` iterations) is tried; then
IG resumes. ``phase3="tabu"`` restores the originally pre-registered TabuCol-only Phase 3
(ablation A6) and ``phase3="ig"`` keeps only the IG stage.

Ablation switches (A1-A6) are keyword arguments. Phase 3 is limited by the pre-registered
wall-clock budget T(m); for the determinism gate G3 a run can be *replayed* deterministically by
passing the IG step count, TabuCol iteration count and polishing step count recorded by the timed
run (``ig_budget``, ``iteration_budget``, ``polish_steps``), which removes the only source of
run-to-run variation (where the clock stops the search).
Original code written for this project.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

from .checker import check_grouping, groups_from_colours
from .colouring import dsatur, greedy_clique, greedy_first_fit, num_colours, renumber
from .conflicts import ConflictGraph
from .iterated_greedy import RULES, ig_step
from .metrics import rhat
from .pauli import PauliSet, _check_relation
from .polish import polish, shot_proxy_S
from .tabucol import TabuParams, reduce_one


IG_STALL = 300        # IG steps without improvement before a TabuCol round (tuning set)
TABU_CAP = 50_000     # TabuCol iterations per round in the hybrid Phase 3 (tuning set)
STALL_BACKOFF = False # back-off variant tried on the tuning set and not adopted
DEFAULT_TABU = TabuParams(tenure_L=50, tenure_lambda=0.6)   # tenure rule after tuning (tuning set)
PHASE3_MODES = ("hybrid", "tabu", "ig")


def default_budget(m: int) -> float:
    """Pre-registered Phase 3 budget T(m) = min(300 s, max(10 s, 0.02 m s))."""
    return float(min(300.0, max(10.0, 0.02 * m)))


@dataclass
class DTPResult:
    """Outcome of one DTP run (all quantities refer to the cleaned PauliSet)."""

    relation: str
    m: int
    n: int
    k: int
    lb: int
    k_seed: int
    k_after_reduce: int
    rhat: float
    rhat_before_polish: float
    rhat_seed: float
    S: float
    S_before_polish: float
    valid: bool
    check_message: str
    optimal_certified: bool
    phase3: str
    rounds: int
    tabu_iterations: int
    tabu_successes: int
    ig_steps: int
    ig_improvements: int
    ig_rule_counts: dict
    stop_reason: str
    polish_passes: int
    polish_moves: int
    polish_steps: int
    polish_stopped: str
    budget_s: float
    timings: dict
    cpu_timings: dict
    trajectory: list
    graph_mode: str
    graph_bytes: int
    gamma: str
    gamma_bytes_peak: int
    density: float
    peak_traced_bytes: int
    groups: list = field(repr=False, default_factory=list)           # indices into PauliSet
    original_groups: list = field(repr=False, default_factory=list)  # indices into the input list
    colours: list = field(repr=False, default_factory=list)          # final colour per term

    def summary(self) -> dict:
        """All scalar fields (groups and colours removed), suitable for a results row."""
        d = asdict(self)
        for key in ("groups", "original_groups", "colours"):
            d.pop(key)
        return d


class _Clock:
    """Wall-clock (perf_counter) and CPU (process_time) stopwatch for one phase."""

    def __init__(self):
        self.w = time.perf_counter()
        self.c = time.process_time()

    def stop(self) -> tuple[float, float]:
        return time.perf_counter() - self.w, time.process_time() - self.c


def run_dtp(pset: PauliSet, relation: str = "QWC", budget: float | None = None, seed: int = 0,
            seed_method: str = "dsatur", clique_stop: bool = True, do_polish: bool = True,
            tabu_params: TabuParams | None = None, polish_passes: int = 20,
            polish_time: float = 60.0, graph_mode: str = "auto", clique_starts: int = 20,
            trace_memory: bool = False, graph: ConflictGraph | None = None, gamma: str = "lazy",
            phase3: str = "hybrid", ig_stall: int = IG_STALL, tabu_cap: int = TABU_CAP,
            stall_backoff: bool = STALL_BACKOFF, iteration_budget: int | None = None, ig_budget: int | None = None,
            polish_steps: int | None = None) -> DTPResult:
    """Run DTP on a cleaned :class:`PauliSet` and return a :class:`DTPResult`.

    Parameters mirror the pre-registered protocol and its amendment:

    * ``budget`` - Phase 3 wall-clock seconds (default T(m));
    * ``seed`` - RNG seed (IG class orders, TabuCol tie-breaking and tenure; A1 random seeding);
    * ``seed_method`` - 'dsatur' (DTP) or 'random' (ablation A1, random-order greedy);
    * ``clique_stop`` - stop Phase 3 when k = LB (ablation A2 sets False);
    * ``gamma`` - 'lazy' (DTP) or 'full' (ablation A3);
    * ``do_polish`` - Phase 4 on/off (ablation A4);
    * ``tabu_params`` - tenure rule (default ``DEFAULT_TABU``; A5 uses ``TabuParams(fixed_tenure=10)``);
    * ``phase3`` - 'hybrid' (DTP), 'tabu' (A6: TabuCol-only Phase 3 as first pre-registered) or 'ig';
    * ``ig_stall`` / ``tabu_cap`` - IG steps without improvement before a TabuCol round, and the
      iteration cap of a TabuCol round in the hybrid; with ``stall_backoff`` the stall threshold
      doubles after every failed TabuCol round and resets after a successful one;
    * ``ig_budget`` / ``iteration_budget`` / ``polish_steps`` - deterministic replay of a timed run
      (G3); when given, the clock limits are not used;
    * ``trace_memory`` - record the tracemalloc peak (slows Python allocations; off by default).
    """
    rel = _check_relation(relation)
    if phase3 not in PHASE3_MODES:
        raise ValueError(f"phase3 must be one of {PHASE3_MODES}")
    params = tabu_params if tabu_params is not None else DEFAULT_TABU
    rng = np.random.default_rng(seed)
    timings: dict[str, float] = {}
    cpu: dict[str, float] = {}
    if trace_memory:
        tracemalloc.start()
    t_all = _Clock()
    m = pset.m
    if m == 0:
        if trace_memory:
            tracemalloc.stop()
        return DTPResult(rel, 0, pset.n, 0, 0, 0, 0, float("nan"), float("nan"), float("nan"), 0.0, 0.0,
                         True, "valid", True, phase3, 0, 0, 0, 0, 0, {}, "empty", 0, 0, 0, "empty", 0.0,
                         {"total": 0.0}, {"total": 0.0}, [], "none", 0, gamma, 0, 0.0, 0, [], [], [])
    # Phase 0/1: conflict graph + seed colouring
    clk = _Clock()
    if graph is None:
        graph = ConflictGraph(pset, rel, mode=graph_mode)
    timings["build"], cpu["build"] = clk.stop()
    clk = _Clock()
    if seed_method == "dsatur":
        col = dsatur(graph)
    elif seed_method == "random":
        col = greedy_first_fit(graph, rng.permutation(m))
    else:
        raise ValueError("seed_method must be 'dsatur' or 'random'")
    timings["seed"], cpu["seed"] = clk.stop()
    col = renumber(col)
    k_seed = num_colours(col)
    rhat_seed = rhat(groups_from_colours(col), pset.coeffs)
    # Phase 2: lower bound
    clk = _Clock()
    lb, _ = greedy_clique(graph, starts=clique_starts)
    timings["bound"], cpu["bound"] = clk.stop()
    # Phase 3: IG / TabuCol alternation
    budget = default_budget(m) if budget is None else float(budget)
    replay = iteration_budget is not None or ig_budget is not None
    ig_cap_total = int(ig_budget or 0)
    tabu_cap_total = int(iteration_budget or 0)
    clk = _Clock()
    deadline = math.inf if replay else clk.w + budget
    use_ig = phase3 in ("hybrid", "ig")
    use_tabu = phase3 in ("hybrid", "tabu")
    k = k_seed
    best = col
    trajectory = [(0.0, k)]
    target = lb if clique_stop else 1
    rounds = successes = its_total = ig_steps = ig_impr = since = 0
    stall = int(ig_stall)
    rule_counts = [0, 0, 0]
    gamma_peak = 0
    stop_reason = "lb_reached" if clique_stop else "target_reached"
    while k > target:
        if not replay and time.perf_counter() >= deadline:
            stop_reason = "deadline"
            break
        if use_ig and (not use_tabu or since < stall):
            if replay and ig_steps >= ig_cap_total:
                stop_reason = "replay_budget"
                break
            best = ig_step(graph, best, k, rng, rule_counts)
            ig_steps += 1
            k_new = num_colours(best)
            if k_new < k:
                k = k_new
                since = 0
                ig_impr += 1
                trajectory.append((time.perf_counter() - clk.w, k))
            else:
                since += 1
            continue
        cap = tabu_cap if phase3 == "hybrid" else None
        if replay:
            left = tabu_cap_total - its_total
            if left <= 0:
                stop_reason = "replay_budget"
                break
            cap = left if cap is None else min(cap, left)
        st: dict = {}
        ok, h, its = reduce_one(graph, best, k, deadline, rng, params, max_iter=cap, gamma=gamma, stats=st)
        rounds += 1
        its_total += its
        gamma_peak = max(gamma_peak, st.get("gamma_bytes", 0))
        since = 0
        if ok:
            best, k = renumber(h), k - 1
            successes += 1
            stall = int(ig_stall)
            trajectory.append((time.perf_counter() - clk.w, k))
        elif stall_backoff:
            stall *= 2
        if not ok and phase3 == "tabu":      # pre-registered v1: a failed round (clock or replay cap) ends Phase 3
            stop_reason = "replay_budget" if replay else "deadline"
            break
    timings["reduce"], cpu["reduce"] = clk.stop()
    k_after_reduce = k
    best = renumber(best)
    S_before = shot_proxy_S(best, pset.coeffs)
    rhat_before = rhat(groups_from_colours(best), pset.coeffs)
    # Phase 4: polishing
    clk = _Clock()
    pstats = {"passes": 0, "moves": 0, "steps": 0, "stopped": "off"}
    if do_polish:
        best, pstats = polish(graph, best, pset.coeffs, max_passes=polish_passes,
                              time_limit=polish_time, max_steps=polish_steps)
    timings["polish"], cpu["polish"] = clk.stop()
    # Phase 5: independent verification
    clk = _Clock()
    groups = groups_from_colours(best)
    ok, msg = check_grouping(pset.strings, groups, rel)
    timings["check"], cpu["check"] = clk.stop()
    timings["total"], cpu["total"] = t_all.stop()
    peak = 0
    if trace_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    original_groups = ([[i for j in g for i in pset.original_indices[j]] for g in groups]
                       if pset.original_indices else [])
    k_final = len(groups)
    return DTPResult(
        relation=rel, m=m, n=pset.n, k=k_final, lb=lb, k_seed=k_seed, k_after_reduce=k_after_reduce,
        rhat=rhat(groups, pset.coeffs), rhat_before_polish=rhat_before, rhat_seed=rhat_seed,
        S=shot_proxy_S(best, pset.coeffs), S_before_polish=S_before, valid=ok, check_message=msg,
        optimal_certified=(k_final == lb), phase3=phase3, rounds=rounds, tabu_iterations=its_total,
        tabu_successes=successes, ig_steps=ig_steps, ig_improvements=ig_impr,
        ig_rule_counts=dict(zip(RULES, rule_counts)), stop_reason=stop_reason,
        polish_passes=int(pstats["passes"]), polish_moves=int(pstats["moves"]),
        polish_steps=int(pstats["steps"]), polish_stopped=str(pstats["stopped"]), budget_s=budget,
        timings=timings, cpu_timings=cpu, trajectory=trajectory, graph_mode=graph.mode,
        graph_bytes=graph.nbytes, gamma=gamma, gamma_bytes_peak=int(gamma_peak),
        density=graph.density, peak_traced_bytes=int(peak), groups=groups,
        original_groups=original_groups, colours=best.tolist())


def group_terms(terms: Sequence[str], coeffs: Sequence[float] | None = None, relation: str = "QWC",
                **kwargs) -> DTPResult:
    """Convenience API: validate raw terms and run DTP (see :func:`run_dtp` for options)."""
    pset = PauliSet.from_terms(terms, coeffs)
    return run_dtp(pset, relation, **kwargs)
