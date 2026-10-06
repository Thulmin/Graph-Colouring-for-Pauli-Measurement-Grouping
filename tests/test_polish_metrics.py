"""Phase 4 and metric tests: polishing and R-hat (T17, T18)."""
from __future__ import annotations

import numpy as np
import pytest

from dtp import QWC, FC, ConflictGraph, PauliSet, check_grouping, groups_from_colours, qwc_basis, rhat
from dtp.colouring import dsatur, greedy_first_fit, num_colours
from dtp.polish import polish, shot_proxy_S

from conftest import random_strings


def example_12_2():
    ps = PauliSet.from_terms(["ZI", "IZ", "XI", "IX"], [0.9, 0.1, 0.1, 0.8])
    g = ConflictGraph(ps, QWC)
    start = np.array([0, 1, 1, 2])       # groups {ZI}, {IZ, XI}, {IX}
    return ps, g, start


# T18
def test_rhat_and_polish_reproduce_dossier_example():
    ps, g, start = example_12_2()
    assert shot_proxy_S(start, ps.coeffs) == pytest.approx(1.8414, abs=1e-4)
    assert rhat(groups_from_colours(start), ps.coeffs) == pytest.approx(1.065, abs=1e-3)
    col, st = polish(g, start, ps.coeffs)
    assert num_colours(col) == 2
    groups = sorted(sorted(ps.strings[i] for i in grp) for grp in groups_from_colours(col))
    assert groups == [["IX", "ZI"], ["IZ", "XI"]]
    assert st["S_after"] == pytest.approx(1.3456, abs=1e-4)
    assert rhat(groups_from_colours(col), ps.coeffs) == pytest.approx(1.994, abs=1e-3)


def test_rhat_singletons_equal_one_and_empty_is_nan():
    a = np.array([0.3, -0.2, 0.7])
    assert rhat([[0], [1], [2]], a) == pytest.approx(1.0)
    assert np.isnan(rhat([], a))


# T17
@pytest.mark.parametrize("rel", [QWC, FC])
def test_polish_keeps_validity_and_never_increases_S_or_k(rel):
    rng = np.random.default_rng(20)
    for trial in range(10):
        strings = list(dict.fromkeys(random_strings(rng, 90, 6)))
        a = rng.normal(size=len(strings)) * rng.choice([0.01, 1.0], size=len(strings))
        ps = PauliSet.from_terms(strings, a)
        g = ConflictGraph(ps, rel)
        col0 = greedy_first_fit(g, rng.permutation(ps.m))
        col, st = polish(g, col0, ps.coeffs)
        ok, msg = check_grouping(ps.strings, groups_from_colours(col), rel)
        assert ok, msg
        assert num_colours(col) <= num_colours(col0)
        assert st["S_after"] <= st["S_before"] + 1e-12
        assert st["stopped"] in ("converged", "max_passes")


def test_polish_step_budget_replays_time_limited_run():
    rng = np.random.default_rng(21)
    strings = list(dict.fromkeys(random_strings(rng, 200, 7)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    g = ConflictGraph(ps, QWC)
    col0 = dsatur(g)
    full, st_full = polish(g, col0, ps.coeffs)
    part, st_part = polish(g, col0, ps.coeffs, max_steps=137)
    assert st_part["steps"] == 137 and st_part["stopped"] == "steps"
    again, st_again = polish(g, col0, ps.coeffs, max_steps=137)
    assert np.array_equal(part, again)
    replay, st_replay = polish(g, col0, ps.coeffs, max_steps=st_full["steps"])
    assert np.array_equal(replay, full)
    stopped, st_t = polish(g, col0, ps.coeffs, time_limit=-1.0)
    assert st_t["stopped"] == "time" and st_t["steps"] == 0 and np.array_equal(stopped, col0)


def test_polish_empty():
    ps = PauliSet.from_terms([], [])
    g = ConflictGraph(ps, QWC)
    col, st = polish(g, np.zeros(0, dtype=np.int64), ps.coeffs)
    assert col.size == 0 and st["moves"] == 0
    assert shot_proxy_S(np.zeros(0, dtype=np.int64), ps.coeffs) == 0.0


def test_qwc_basis():
    assert qwc_basis(["XIZ", "XYI", "IYZ"]) == "XYZ"
    assert qwc_basis(["ZI", "ZI"]) == "Z*"
    with pytest.raises(ValueError):
        qwc_basis(["XI", "ZI"])
