"""Phase 1-2 tests: DSATUR, greedy orders and the clique lower bound (T07, T08, T12-T14)."""
from __future__ import annotations

import numpy as np
import pytest

from dtp import FC, QWC, ConflictGraph, PauliSet, check_grouping, groups_from_colours
from dtp.colouring import (dsatur, greedy_clique, greedy_first_fit, num_colours, renumber,
                           smallest_last_order)

from conftest import brute_force_chi, graph_pset, random_graph, random_strings


def is_proper(graph, col) -> bool:
    return graph.count_monochromatic(np.asarray(col)) == 0


# T07
def test_all_compatible_gives_one_group():
    ps = PauliSet.from_terms(["ZI", "IZ", "ZZ"], [1, 1, 1])
    g = ConflictGraph(ps, QWC)
    assert num_colours(dsatur(g)) == 1


# T08
def test_all_conflicting_gives_m_groups_and_tight_bound():
    ps = PauliSet.from_terms(["XI", "YI", "ZI"], [1, 1, 1])
    g = ConflictGraph(ps, QWC)
    assert num_colours(dsatur(g)) == 3
    lb, members = greedy_clique(g)
    assert lb == 3 and sorted(members.tolist()) == [0, 1, 2]


# T12
@pytest.mark.parametrize("rel", [QWC, FC])
def test_dsatur_proper_on_random_pauli_sets(rel):
    rng = np.random.default_rng(10)
    for trial in range(15):
        strings = list(dict.fromkeys(random_strings(rng, 120, 8)))
        ps = PauliSet.from_terms(strings, [1.0] * len(strings))
        g = ConflictGraph(ps, rel)
        col = dsatur(g)
        assert is_proper(g, col)
        ok, msg = check_grouping(ps.strings, groups_from_colours(col), rel)
        assert ok, msg
        assert col.min() == 0 and set(np.unique(col)) == set(range(num_colours(col)))


def test_dsatur_proper_on_random_graphs():
    rng = np.random.default_rng(11)
    for trial in range(20):
        nv = int(rng.integers(5, 30))
        edges = random_graph(rng, nv, float(rng.uniform(0.1, 0.8)))
        g = ConflictGraph(graph_pset(nv, edges), QWC)
        assert is_proper(g, dsatur(g))


# T13
@pytest.mark.parametrize("nv", [4, 6, 10, 20])
def test_dsatur_exact_on_even_cycles(nv):
    edges = [(i, (i + 1) % nv) for i in range(nv)]
    g = ConflictGraph(graph_pset(nv, edges), FC)
    assert num_colours(dsatur(g)) == 2


def test_dsatur_exact_on_trees():
    rng = np.random.default_rng(12)
    for nv in (2, 7, 15, 40):
        edges = [(int(rng.integers(0, v)), v) for v in range(1, nv)]
        g = ConflictGraph(graph_pset(nv, edges), QWC)
        assert num_colours(dsatur(g)) == 2


# T14
def test_clique_bound_never_exceeds_chromatic_number():
    rng = np.random.default_rng(13)
    for trial in range(25):
        nv = int(rng.integers(2, 8))
        edges = random_graph(rng, nv, float(rng.uniform(0.2, 0.9)))
        g = ConflictGraph(graph_pset(nv, edges), QWC)
        lb, members = greedy_clique(g)
        chi = brute_force_chi(nv, edges)
        assert 1 <= lb <= chi <= num_colours(dsatur(g))
        mem = members.tolist()                       # returned members form a clique
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                assert g.conflicts(mem[a], mem[b])


def test_clique_static_fallback_for_packed_graphs():
    rng = np.random.default_rng(14)
    edges = random_graph(rng, 25, 0.5)
    ps = graph_pset(25, edges)
    lb_packed, mem = greedy_clique(ConflictGraph(ps, QWC, mode="packed"))
    g = ConflictGraph(ps, QWC)
    assert 1 <= lb_packed <= num_colours(dsatur(g))
    for a in range(len(mem)):
        for b in range(a + 1, len(mem)):
            assert g.conflicts(int(mem[a]), int(mem[b]))


def test_empty_graph_helpers():
    ps = PauliSet.from_terms([], [])
    g = ConflictGraph(ps, QWC)
    assert dsatur(g).size == 0
    assert greedy_clique(g)[0] == 0
    assert num_colours(np.zeros(0, dtype=np.int64)) == 0
    assert renumber(np.zeros(0, dtype=np.int64)).size == 0


def test_greedy_orders_proper():
    rng = np.random.default_rng(15)
    edges = random_graph(rng, 40, 0.3)
    g = ConflictGraph(graph_pset(40, edges), QWC)
    order = smallest_last_order(g)
    assert sorted(order.tolist()) == list(range(40))
    for o in (order, rng.permutation(40), np.arange(40)):
        assert is_proper(g, greedy_first_fit(g, o))


def test_renumber_preserves_order():
    assert renumber(np.array([5, 2, 5, 9])).tolist() == [1, 0, 1, 2]
