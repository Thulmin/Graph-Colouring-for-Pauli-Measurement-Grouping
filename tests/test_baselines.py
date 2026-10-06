"""Baseline adapters and heuristics (T23, T24 and correctness of the classical baselines)."""
from __future__ import annotations

import numpy as np
import pytest

from dtp import FC, QWC, ConflictGraph, PauliSet, check_grouping, groups_from_colours, letters_compatible
from dtp import baselines as B
from dtp.colouring import dsatur, greedy_clique, num_colours
from dtp.instances import load

from conftest import SMOKE_INSTANCE, brute_force_chi, graph_pset, random_graph, random_strings, requires_smoke


def naive_sorted_insertion(strings, coeffs, rel):
    order = sorted(range(len(strings)), key=lambda i: -abs(coeffs[i]))   # stable
    groups: list[list[int]] = []
    for i in order:
        for g in groups:
            if all(letters_compatible(strings[i], strings[j], rel) for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def valid(ps, col, rel):
    ok, msg = check_grouping(ps.strings, groups_from_colours(col), rel)
    return ok


@pytest.fixture(scope="module")
def small_set():
    rng = np.random.default_rng(40)
    strings = list(dict.fromkeys(random_strings(rng, 160, 8)))
    return PauliSet.from_terms(strings, rng.normal(size=len(strings)) * rng.choice([0.1, 1.0], len(strings)))


@pytest.mark.parametrize("rel", [QWC, FC])
def test_sorted_insertion_matches_naive_definition(small_set, rel):
    g = ConflictGraph(small_set, rel)
    out = B.sorted_insertion(g, small_set.coeffs)
    naive = naive_sorted_insertion(small_set.strings, small_set.coeffs, rel)
    assert sorted(map(sorted, groups_from_colours(out.colours))) == sorted(map(sorted, naive))


@pytest.mark.parametrize("rel", [QWC, FC])
def test_classical_baselines_valid(small_set, rel):
    g = ConflictGraph(small_set, rel)
    for out in (B.random_greedy(g, 0), B.random_greedy(g, 1), B.smallest_last(g)):
        assert valid(small_set, out.colours, rel)
    a = B.random_greedy(g, 5).colours
    b = B.random_greedy(g, 5).colours
    assert np.array_equal(a, b)


@pytest.mark.parametrize("rel", [QWC, FC])
def test_iterated_greedy_monotone_and_valid(small_set, rel):
    g = ConflictGraph(small_set, rel)
    col0 = B.random_greedy(g, 3).colours
    out = B.iterated_greedy(g, col0, budget=60.0, seed=0, max_iter=200)
    assert valid(small_set, out.colours, rel)
    assert num_colours(out.colours) <= num_colours(col0)
    ks = [k for _, k in out.extra["trajectory"]]
    assert ks == sorted(ks, reverse=True)
    assert out.extra["iterations"] <= 200
    lb, _ = greedy_clique(g)
    stopped = B.iterated_greedy(g, col0, budget=60.0, seed=0, lb=num_colours(col0))
    assert stopped.extra["iterations"] == 0


def test_gcol_adapter_valid_and_seeded(small_set):
    pytest.importorskip("gcol")
    g = ConflictGraph(small_set, QWC)
    a = B.gcol_colouring(g, 2, 500, seed=0)
    b = B.gcol_colouring(g, 2, 500, seed=0)
    c = B.gcol_colouring(g, 3, 500, seed=0)
    assert valid(small_set, a.colours, QWC) and valid(small_set, c.colours, QWC)
    assert np.array_equal(a.colours, b.colours)


def test_cpsat_finds_brute_force_optimum():
    pytest.importorskip("ortools")
    rng = np.random.default_rng(41)
    for trial in range(6):
        nv = int(rng.integers(4, 8))
        edges = random_graph(rng, nv, 0.5)
        ps = graph_pset(nv, edges)
        g = ConflictGraph(ps, QWC)
        ub = dsatur(g)
        _, clique = greedy_clique(g)
        out = B.cpsat_exact(g, ub, clique, time_limit=20, workers=1)
        assert out.extra["status"] == "OPTIMAL"
        assert num_colours(out.colours) == out.extra["objective"] == brute_force_chi(nv, edges)
        assert valid(ps, out.colours, QWC)


def test_colours_from_groups_errors():
    with pytest.raises(ValueError):
        B.colours_from_groups([[0], [0, 1]], 2)
    with pytest.raises(ValueError):
        B.colours_from_groups([[0]], 2)


# T23
@requires_smoke
@pytest.mark.parametrize("rel", [QWC, FC])
def test_qiskit_adapter_matches_qiskit_call(rel):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import PauliList
    inst = load(SMOKE_INSTANCE)
    ps = PauliSet.from_terms(inst.terms, inst.coeffs)
    out = B.qiskit_lf(ps, rel)
    assert valid(ps, out.colours, rel)
    native = PauliList([s[::-1] for s in ps.strings]).group_commuting(qubit_wise=(rel == QWC))
    assert num_colours(out.colours) == len(native)


# T24
@requires_smoke
@pytest.mark.parametrize("method", ["lf", "dsatur", "rlf", "gis"])
def test_pennylane_adapters_valid(method):
    pytest.importorskip("pennylane")
    inst = load(SMOKE_INSTANCE)
    ps = PauliSet.from_terms(inst.terms, inst.coeffs)
    for rel in (QWC, FC):
        out = B.pennylane_groups(ps, rel, method)
        assert valid(ps, out.colours, rel)
