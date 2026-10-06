"""Constructive colouring and bounding components of DTP (Phases 1 and 2).

* :func:`dsatur`         - DSATUR (Brelaz, 1979) with deterministic tie-breaking.
* :func:`greedy_first_fit` - First-Fit greedy in a given vertex order (used by baselines and
                            Iterated Greedy).
* :func:`smallest_last_order` - Smallest-Last vertex order (Matula & Beck, 1983).
* :func:`greedy_clique`  - greedy clique lower bound (dynamic candidate-degree rule, several
                            starts; static-degree fallback for non-dense graphs).

All functions work on any :class:`~dtp.conflicts.ConflictGraph` representation.
Original code written for this project (algorithms attributed in docstrings).
"""
from __future__ import annotations

import numpy as np

from .conflicts import ConflictGraph


def dsatur(graph: ConflictGraph) -> np.ndarray:
    """DSATUR colouring (Brelaz, 1979).

    Repeatedly selects the uncoloured vertex with the largest saturation (number of
    distinct colours among coloured neighbours), breaking ties by larger degree and then
    by lower index, and gives it the smallest colour not used by its coloured neighbours.

    Returns an int64 array of colours 0..k-1. Time Theta(m^2) with dense rows.
    """
    m = graph.m
    col = np.full(m, -1, dtype=np.int64)
    if m == 0:
        return col
    deg = graph.degrees.astype(np.int64)
    scale = int(deg.max()) + 1
    sat = np.zeros(m, dtype=np.int64)
    uncoloured = np.ones(m, dtype=bool)
    nbytes = (m + 7) // 8
    classmask: list[np.ndarray] = []        # packed: vertices adjacent to colour class c
    for _ in range(m):
        key = np.where(uncoloured, sat * scale + deg, -1)
        v = int(np.argmax(key))             # first maximum -> lowest index on ties
        row = graph.row(v)
        nbc = col[row]
        used = np.bincount(nbc[nbc >= 0], minlength=len(classmask) + 1)
        c = int(np.flatnonzero(used == 0)[0])
        col[v] = c
        uncoloured[v] = False
        if c == len(classmask):
            classmask.append(np.zeros(nbytes, dtype=np.uint8))
        already = np.unpackbits(classmask[c], count=m).astype(bool)
        newly = row & uncoloured & ~already
        sat[newly] += 1
        classmask[c] |= np.packbits(row)
    return col


def greedy_first_fit(graph: ConflictGraph, order: np.ndarray) -> np.ndarray:
    """First-Fit greedy colouring visiting vertices in ``order`` (Welsh-Powell when the order
    is by decreasing degree; random-order greedy when the order is a random permutation)."""
    m = graph.m
    col = np.full(m, -1, dtype=np.int64)
    for v in order:
        nbc = col[graph.row(int(v))]
        nbc = nbc[nbc >= 0]
        if nbc.size == 0:
            col[v] = 0
            continue
        used = np.bincount(nbc, minlength=int(nbc.max()) + 2)
        col[v] = int(np.flatnonzero(used == 0)[0])
    return col


def smallest_last_order(graph: ConflictGraph) -> np.ndarray:
    """Smallest-last vertex order (Matula & Beck, 1983): repeatedly remove a vertex of minimum
    remaining degree; colour in the reverse of the removal order."""
    m = graph.m
    deg = graph.degrees.astype(np.int64).copy()
    alive = np.ones(m, dtype=bool)
    removal = np.empty(m, dtype=np.int64)
    for t in range(m):
        key = np.where(alive, deg, np.iinfo(np.int64).max)
        v = int(np.argmin(key))
        removal[t] = v
        alive[v] = False
        deg[graph.row(v) & alive] -= 1
    return removal[::-1].copy()


def greedy_clique(graph: ConflictGraph, starts: int = 20, work_cap: float = 3e8) -> tuple[int, np.ndarray]:
    """Greedy clique lower bound.

    From each of up to ``starts`` highest-degree vertices, repeatedly add the candidate with
    the most neighbours *inside the current candidate set* (dynamic rule; candidate degrees are
    updated incrementally, total O(|cand|^2) per start). If the dense matrix is unavailable or the
    estimated work exceeds ``work_cap``, the cheaper static-degree rule is used instead.
    Returns (size, members) of the largest clique found; size <= omega(G) <= chi(G).

    Design note (logged 1 Oct 2026, before evaluation runs): the dynamic rule replaced the
    static rule of the dossier pseudocode after the tuning molecule HF showed a weaker bound
    (132 vs 138) with the static rule.
    """
    m = graph.m
    if m == 0:
        return 0, np.zeros(0, dtype=np.int64)
    deg = graph.degrees
    best_members = np.array([int(np.argmax(deg))])
    dense = graph._dense if graph.mode == "dense" else None
    order = np.argsort(-deg, kind="stable")
    c0 = int(deg[order[0]])
    if dense is not None and c0 > 0:
        starts = int(max(1, min(starts, work_cap // max(1, c0 * c0))))
    for v0 in order[:starts]:
        members = [int(v0)]
        idx = graph.neighbours(int(v0))
        if dense is not None:
            dc = dense[np.ix_(idx, idx)].sum(axis=1)
            while idx.size:
                p = int(np.argmax(dc))
                u = int(idx[p])
                members.append(u)
                keep = dense[u, idx]
                removed = idx[~keep]
                idx = idx[keep]
                dc = dc[keep]
                if idx.size and removed.size:
                    dc = dc - dense[np.ix_(idx, removed)].sum(axis=1)
        else:
            cand = graph.row(int(v0)).copy()
            while cand.any():
                ids = np.flatnonzero(cand)
                u = int(ids[np.argmax(deg[ids])])
                members.append(u)
                cand &= graph.row(u)
        if len(members) > best_members.size:
            best_members = np.array(members)
    return int(best_members.size), best_members


def num_colours(col: np.ndarray) -> int:
    """Number of colours of a colouring with colours 0..k-1 (0 for an empty colouring)."""
    return 0 if col.size == 0 else int(col.max()) + 1


def renumber(col: np.ndarray) -> np.ndarray:
    """Map used colours to 0..k-1 preserving their relative order."""
    if col.size == 0:
        return col
    used = np.unique(col)
    lut = np.full(int(used.max()) + 1, -1, dtype=np.int64)
    lut[used] = np.arange(used.size)
    return lut[col]
