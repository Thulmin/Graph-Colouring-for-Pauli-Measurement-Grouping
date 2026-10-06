"""Baseline grouping methods (pre-registration, section 6).

Deployed software defaults (called through their public APIs, versions pinned in pyproject):

* :func:`qiskit_lf`         - ``qiskit.quantum_info.PauliList.group_commuting`` (rustworkx
                              ``graph_greedy_color`` with its default largest-degree-first strategy);
* :func:`pennylane_groups`  - ``pennylane.pauli.compute_partition_indices`` with method
                              'lf' (default), 'dsatur', 'rlf' or 'gis'.

Classical heuristics implemented here on the same conflict graph:

* :func:`random_greedy`, :func:`smallest_last`, :func:`sorted_insertion` (Crawford et al., 2021),
  :func:`iterated_greedy` (Culberson & Luo, 1996; same time budget T(m) as DTP).

Independent implementations and exact optimum:

* :func:`gcol_colouring`    - GCol (Lewis & Palmer) DSATUR + TabuCol (opt_alg=2) or + PartialCol
                              (opt_alg=3) at a fixed iteration limit;
* :func:`cpsat_exact`       - OR-Tools CP-SAT assignment model with clique fixing.

Every function returns colours (one int per term, 0..k-1) for the cleaned PauliSet order.
Original adapter and heuristic code written for this project; the libraries named above are
third-party (see README for licences and versions).
"""
from __future__ import annotations

import random as _pyrandom
import time
from dataclasses import dataclass, field

import numpy as np

from .colouring import greedy_first_fit, renumber, smallest_last_order
from .conflicts import ConflictGraph
from .iterated_greedy import RULES, ig_step
from .pauli import FC, QWC, PauliSet, _check_relation


@dataclass
class BaselineOutput:
    """Colours plus method-specific statistics (seconds measured around the core call)."""

    colours: np.ndarray
    seconds: float
    cpu_seconds: float
    extra: dict = field(default_factory=dict)


def _timed(fn, *args, **kwargs):
    w, c = time.perf_counter(), time.process_time()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - w, time.process_time() - c


def colours_from_groups(groups, m: int) -> np.ndarray:
    """Convert a list of index groups into a colour array; raises if a term is missing or repeated."""
    col = np.full(m, -1, dtype=np.int64)
    for g, members in enumerate(groups):
        for i in members:
            if col[i] != -1:
                raise ValueError(f"term {i} appears in more than one group")
            col[i] = g
    if (col < 0).any():
        raise ValueError(f"{int((col < 0).sum())} term(s) missing from the grouping")
    return col


# --------------------------------------------------------------------------- deployed defaults
def qiskit_lf(pset: PauliSet, relation: str) -> BaselineOutput:
    """Qiskit ``PauliList.group_commuting(qubit_wise=...)`` mapped back to term indices.

    Qiskit labels are big-endian (rightmost character = qubit 0) while our strings put qubit 0
    first, so each string is reversed; this relabelling of qubits preserves every commutation
    relation. Strings in a PauliSet are unique, so labels map back to indices unambiguously.
    """
    from qiskit.quantum_info import PauliList
    rel = _check_relation(relation)
    t0 = time.perf_counter()
    plist = PauliList([s[::-1] for s in pset.strings])
    convert = time.perf_counter() - t0
    groups, sec, cpu = _timed(plist.group_commuting, qubit_wise=(rel == QWC))
    index = {s: i for i, s in enumerate(pset.strings)}
    idx_groups = [[index[lab[::-1]] for lab in g.to_labels()] for g in groups]
    return BaselineOutput(colours_from_groups(idx_groups, pset.m), sec, cpu,
                          {"convert_seconds": convert})


def pennylane_groups(pset: PauliSet, relation: str, method: str = "lf") -> BaselineOutput:
    """PennyLane ``compute_partition_indices`` with grouping type 'qwc' (QWC) or
    'commuting' (FC) and colouring method 'lf', 'dsatur', 'rlf' or 'gis'."""
    import pennylane as qml
    from pennylane.pauli import compute_partition_indices, string_to_pauli_word
    rel = _check_relation(relation)
    t0 = time.perf_counter()
    with qml.QueuingManager.stop_recording():
        obs = [string_to_pauli_word(s) for s in pset.strings]
    convert = time.perf_counter() - t0
    gtype = "qwc" if rel == QWC else "commuting"
    parts, sec, cpu = _timed(compute_partition_indices, obs, grouping_type=gtype, method=method)
    return BaselineOutput(colours_from_groups(parts, pset.m), sec, cpu, {"convert_seconds": convert})


# --------------------------------------------------------------------------- classical heuristics
def random_greedy(graph: ConflictGraph, seed: int) -> BaselineOutput:
    """First-fit greedy in a uniformly random vertex order (numpy Generator seeded by ``seed``)."""
    rng = np.random.default_rng(seed)
    col, sec, cpu = _timed(lambda: greedy_first_fit(graph, rng.permutation(graph.m)))
    return BaselineOutput(col, sec, cpu)


def smallest_last(graph: ConflictGraph) -> BaselineOutput:
    """First-fit greedy in smallest-last order (Matula & Beck, 1983)."""
    col, sec, cpu = _timed(lambda: greedy_first_fit(graph, smallest_last_order(graph)))
    return BaselineOutput(col, sec, cpu)


def sorted_insertion(graph: ConflictGraph, coeffs: np.ndarray) -> BaselineOutput:
    """Sorted Insertion (Crawford et al., 2021): terms in order of decreasing |coefficient|
    (stable for ties), each placed in the first existing group it is compatible with, else in a
    new group. This is exactly first-fit greedy colouring in that order."""
    order = np.argsort(-np.abs(np.asarray(coeffs)), kind="stable")
    col, sec, cpu = _timed(greedy_first_fit, graph, order)
    return BaselineOutput(col, sec, cpu)


def iterated_greedy(graph: ConflictGraph, col0: np.ndarray, budget: float, seed: int,
                    lb: int | None = None, max_iter: int | None = None) -> BaselineOutput:
    """Iterated Greedy (Culberson & Luo, 1996) started from a proper colouring ``col0``.

    Repeats :func:`dtp.iterated_greedy.ig_step` (class-preserving greedy re-colouring, which can
    never use more colours) until the time ``budget`` (seconds) ends, ``max_iter`` steps have been
    made, or k reaches ``lb``. With the same seed, start colouring and budget this is exactly DTP
    with its TabuCol stage removed (and without polishing).
    """
    rng = np.random.default_rng(seed)
    w0, c0 = time.perf_counter(), time.process_time()
    col = renumber(np.asarray(col0, dtype=np.int64))
    k = int(col.max()) + 1 if col.size else 0
    deadline = w0 + budget
    it = 0
    trajectory = [(0.0, k)]
    rules = [0, 0, 0]
    while col.size and time.perf_counter() < deadline:
        if lb is not None and k <= lb:
            break
        if max_iter is not None and it >= max_iter:
            break
        col = ig_step(graph, col, k, rng, rules)
        it += 1
        k_new = int(col.max()) + 1
        if k_new < k:
            trajectory.append((time.perf_counter() - w0, k_new))
        k = k_new
    return BaselineOutput(col, time.perf_counter() - w0, time.process_time() - c0,
                          {"iterations": it, "trajectory": trajectory,
                           "rule_counts": dict(zip(RULES, rules))})


# --------------------------------------------------------------------------- independent implementation
def gcol_colouring(graph: ConflictGraph, opt_alg: int, it_limit: int, seed: int) -> BaselineOutput:
    """GCol ``node_coloring(G, strategy='dsatur', opt_alg, it_limit)`` on a NetworkX copy of the
    conflict graph. GCol draws random numbers from Python's ``random`` module, which is seeded."""
    import gcol
    import networkx as nx
    t0 = time.perf_counter()
    G = nx.Graph()
    G.add_nodes_from(range(graph.m))
    G.add_edges_from(map(tuple, graph.edge_list().tolist()))
    build = time.perf_counter() - t0
    _pyrandom.seed(seed)
    c, sec, cpu = _timed(gcol.node_coloring, G, strategy="dsatur", opt_alg=opt_alg, it_limit=it_limit)
    col = renumber(np.array([c[v] for v in range(graph.m)], dtype=np.int64))
    return BaselineOutput(col, sec, cpu, {"networkx_build_seconds": build, "edges": G.number_of_edges()})


# --------------------------------------------------------------------------- exact optimum
def cpsat_exact(graph: ConflictGraph, ub_colours: np.ndarray, clique: np.ndarray,
                time_limit: float = 600.0, workers: int = 2, seed: int = 0) -> BaselineOutput:
    """Minimum colouring by CP-SAT (assignment model, design dossier section 7).

    Variables x[v,c] (v gets colour c) and w[c] (colour c used) for c < K, where K is the number
    of colours of the feasible colouring ``ub_colours`` (an upper bound only; no solution hint is
    given, so the solver does not start from a heuristic solution). Constraints: one colour per vertex; x[u,c] + x[v,c]
    <= w[c] for every conflict edge; w[c] >= w[c+1] (symmetry breaking); members of ``clique``
    are fixed to colours 0..|Q|-1. Objective: minimise sum_c w[c].
    Returns colours; ``extra`` holds the status, objective, best bound and wall time.
    """
    from ortools.sat.python import cp_model
    m = graph.m
    K = int(np.max(ub_colours)) + 1
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x{v}_{c}") for c in range(K)] for v in range(m)]
    w = [model.NewBoolVar(f"w{c}") for c in range(K)]
    for v in range(m):
        model.AddExactlyOne(x[v])
    for u, v in graph.edge_list().tolist():
        for c in range(K):
            model.Add(x[u][c] + x[v][c] <= w[c])
    for c in range(K - 1):
        model.Add(w[c] >= w[c + 1])
    for i, v in enumerate(np.asarray(clique).tolist()):
        model.Add(x[v][i] == 1)
    model.Minimize(sum(w))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    w0, c0 = time.perf_counter(), time.process_time()
    status = solver.Solve(model)
    sec, cpu = time.perf_counter() - w0, time.process_time() - c0
    name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        col = np.array([next(c for c in range(K) if solver.Value(x[v][c])) for v in range(m)], dtype=np.int64)
        col = renumber(col)
        obj = int(round(solver.ObjectiveValue()))
    else:
        col = np.asarray(ub_colours, dtype=np.int64)
        obj = K
    return BaselineOutput(col, sec, cpu, {"status": name, "objective": obj,
                                          "best_bound": float(solver.BestObjectiveBound()),
                                          "K_upper": K, "clique_size": int(len(clique)),
                                          "workers": int(workers)})
