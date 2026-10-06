"""Shared fixtures and helpers for the test suite (design dossier, section 15, tests T01-T26)."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from dtp import PauliSet
from dtp.pauli import realise_graph

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "instances"


def random_strings(rng: np.random.Generator, m: int, n: int, p_identity: float = 0.4) -> list[str]:
    letters = np.array(list("IXYZ"))
    probs = np.array([p_identity] + [(1 - p_identity) / 3] * 3)
    return ["".join(rng.choice(letters, size=n, p=probs)) for _ in range(m)]


def graph_pset(num_vertices: int, edges) -> PauliSet:
    """PauliSet whose conflict graph (QWC and FC) is the given graph (Lemma 3)."""
    strings = realise_graph(num_vertices, edges)
    return PauliSet.from_terms(strings, [1.0] * num_vertices)


def brute_force_chi(num_vertices: int, edges) -> int:
    """Chromatic number by exhaustive search (tiny graphs only)."""
    if num_vertices == 0:
        return 0
    edges = list(edges)
    for k in range(1, num_vertices + 1):
        for col in itertools.product(range(k), repeat=num_vertices):
            if col[0] != 0:
                continue
            if all(col[u] != col[v] for u, v in edges):
                return k
    return num_vertices


def random_graph(rng: np.random.Generator, nv: int, p: float):
    return [(u, v) for u in range(nv) for v in range(u + 1, nv) if rng.random() < p]


def have_instance(name: str) -> bool:
    return (INSTANCES / f"{name}.json").exists()


# Adapter and end-to-end tests use the TUNING molecule HF (m = 630), never an evaluation molecule,
# so that no evaluation result exists before the pre-registered runs.
SMOKE_INSTANCE = "HF_sto-3g_JW"
requires_smoke = pytest.mark.skipif(not have_instance(SMOKE_INSTANCE),
                                    reason="HF instance not generated (run scripts/generate_instances.py)")
