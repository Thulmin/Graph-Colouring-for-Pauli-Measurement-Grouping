"""Phase 3 tests: TabuCol, k-reduction and implementation equivalence (T15, T16)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dtp import FC, QWC, ConflictGraph, PauliSet
from dtp.colouring import dsatur, greedy_first_fit, num_colours
from dtp.tabucol import TabuParams, reduce_one, tabucol, tabucol_reference

from conftest import graph_pset, random_graph, random_strings


def c6():
    edges = [(i, (i + 1) % 6) for i in range(6)]
    return ConflictGraph(graph_pset(6, edges), QWC)


# T15 (design dossier 12.3: greedy in the order 0,3,1,4,2,5 gives 3 colours on C6)
def test_tabucol_two_colours_c6_from_greedy_start():
    g = c6()
    start = greedy_first_fit(g, np.array([0, 3, 1, 4, 2, 5]))
    assert start.tolist() == [0, 1, 2, 0, 1, 2]
    ok, col, its = reduce_one(g, start, 3, math.inf, np.random.default_rng(0))
    assert ok and num_colours(col) == 2
    assert g.count_monochromatic(col) == 0


@pytest.mark.parametrize("gamma", ["lazy", "full"])
def test_tabucol_bipartite_from_bad_start(gamma):
    rng = np.random.default_rng(1)
    for nv in (8, 12, 30):
        edges = [(i, (i + 1) % nv) for i in range(nv)]
        g = ConflictGraph(graph_pset(nv, edges), FC)
        start = np.zeros(nv, dtype=np.int64)              # every edge conflicting
        ok, col, _ = tabucol(g, start, 2, math.inf, rng, gamma=gamma, max_iter=10_000)
        assert ok and g.count_monochromatic(col) == 0


# T16
@pytest.mark.parametrize("rel", [QWC, FC])
def test_reduce_never_returns_invalid_success(rel):
    rng = np.random.default_rng(2)
    strings = list(dict.fromkeys(random_strings(rng, 150, 7)))
    ps = PauliSet.from_terms(strings, [1.0] * len(strings))
    g = ConflictGraph(ps, rel)
    col = dsatur(g)
    k = num_colours(col)
    for seed in range(8):
        ok, h, its = reduce_one(g, col, k, math.inf, np.random.default_rng(seed), max_iter=2_000)
        assert its <= 2_000
        if ok:
            assert num_colours(h) <= k - 1
            assert g.count_monochromatic(h) == 0
        else:
            assert its == 2_000


@pytest.mark.parametrize("rel", [QWC, FC])
@pytest.mark.parametrize("params", [TabuParams(), TabuParams(fixed_tenure=10), TabuParams(tenure_L=3, tenure_lambda=1.5)])
def test_lazy_full_and_reference_follow_same_search_path(rel, params):
    rng = np.random.default_rng(3)
    strings = list(dict.fromkeys(random_strings(rng, 180, 8, p_identity=0.45)))
    ps = PauliSet.from_terms(strings, [1.0] * len(strings))
    g = ConflictGraph(ps, rel)
    col = dsatur(g)
    k = num_colours(col)
    for kk in (k - 1, max(2, k - 3)):
        start = np.minimum(col, kk - 1)
        outs = []
        for fn, kw in ((tabucol_reference, {}), (tabucol, {"gamma": "lazy"}), (tabucol, {"gamma": "full"})):
            outs.append(fn(g, start, kk, math.inf, np.random.default_rng(11), params, max_iter=2_500, **kw))
        ok0, col0, its0 = outs[0]
        for ok, c, its in outs[1:]:
            assert ok == ok0 and its == its0
            assert np.array_equal(c, col0)


def test_tabucol_respects_deadline_and_round_cap():
    rng = np.random.default_rng(4)
    edges = random_graph(rng, 60, 0.5)
    g = ConflictGraph(graph_pset(60, edges), QWC)
    start = np.zeros(60, dtype=np.int64)
    ok, col, its = tabucol(g, start, 3, deadline=0.0, rng=rng)          # deadline already passed
    assert not ok and its == 0
    ok, col, its = tabucol(g, start, 3, math.inf, rng, TabuParams(max_iter_per_round=50))
    assert its <= 50
    ok, col, its = tabucol_reference(g, start, 3, math.inf, rng, TabuParams(max_iter_per_round=50))
    assert its <= 50


def test_tabucol_rejects_unknown_gamma():
    g = c6()
    with pytest.raises(ValueError):
        tabucol(g, np.zeros(6, dtype=np.int64), 2, math.inf, np.random.default_rng(0), gamma="dense")


def test_tabucol_stats_report_gamma_memory():
    rng = np.random.default_rng(5)
    edges = random_graph(rng, 50, 0.4)
    g = ConflictGraph(graph_pset(50, edges), QWC)
    col = dsatur(g)
    k = num_colours(col)
    st_lazy, st_full = {}, {}
    reduce_one(g, col, k, math.inf, np.random.default_rng(0), max_iter=300, gamma="lazy", stats=st_lazy)
    reduce_one(g, col, k, math.inf, np.random.default_rng(0), max_iter=300, gamma="full", stats=st_full)
    assert st_full["gamma_rows_peak"] == 50
    assert st_lazy["gamma_rows_peak"] <= 50
