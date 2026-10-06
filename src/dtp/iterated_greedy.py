"""Iterated Greedy re-colouring (Culberson & Luo, 1996), shared by DTP Phase 3 and the IG baseline.

One IG step takes a proper colouring, lists the vertices class by class (each colour class kept
contiguous, vertices inside a class in index order) and re-runs first-fit greedy colouring in that
order. Because every class is an independent set and is visited as a block, greedy never needs a
new colour for a later class, so the number of colours can only stay the same or fall
(Culberson & Luo, 1996). The order of the classes is chosen uniformly at random among three rules
named by Culberson & Luo: reverse of the current colour order, largest class first, and random.

Original code written for this project.
"""
from __future__ import annotations

import numpy as np

from .colouring import greedy_first_fit, renumber
from .conflicts import ConflictGraph

RULES = ("reverse", "largest_first", "random")


def ig_step(graph: ConflictGraph, col: np.ndarray, k: int, rng: np.random.Generator,
            counts: list | None = None) -> np.ndarray:
    """One Iterated Greedy step on a proper colouring ``col`` with colours 0..k-1.

    Returns a proper colouring with colours 0..k'-1, k' <= k. ``counts`` (length 3), if given,
    is incremented at the index of the class-order rule used.
    """
    rule = int(rng.integers(3))
    if counts is not None:
        counts[rule] += 1
    if rule == 0:
        class_order = np.arange(k - 1, -1, -1)
    elif rule == 1:
        class_order = np.argsort(-np.bincount(col, minlength=k), kind="stable")
    else:
        class_order = rng.permutation(k)
    rank = np.empty(k, dtype=np.int64)
    rank[class_order] = np.arange(k)
    perm = np.argsort(rank[col], kind="stable")
    return renumber(greedy_first_fit(graph, perm))
