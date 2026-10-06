"""End-to-end tests of the DTP pipeline (T19-T22, determinism replay, ablation switches)."""
from __future__ import annotations

import numpy as np
import pytest

from dtp import FC, QWC, ConflictGraph, PauliSet, check_grouping, group_terms, run_dtp
from dtp.dtp import default_budget
from dtp.tabucol import TabuParams

from conftest import brute_force_chi, random_strings

H2_STO3G_JW = ["IIIZ", "IIZI", "IIZZ", "IZII", "IZIZ", "IZZI", "XXYY", "XYYX", "YXXY", "YYXX",
               "ZIII", "ZIIZ", "ZIZI", "ZZII"]
H2_COEFFS = [-0.222786, -0.222786, 0.174348, 0.171198, 0.120545, 0.165867, -0.045322, 0.045322,
             0.045322, -0.045322, 0.171198, 0.165867, 0.120545, 0.168622]


def test_default_budget():
    assert default_budget(10) == 10.0
    assert default_budget(5000) == 100.0
    assert default_budget(10 ** 6) == 300.0


# T19
def test_known_answer_h2_qwc():
    r = group_terms(H2_STO3G_JW, H2_COEFFS, "QWC", budget=5)
    assert r.k == 5 and r.lb == 5 and r.optimal_certified and r.valid
    assert r.stop_reason == "lb_reached" and r.rounds == 0


def test_known_answer_h2_fc():
    r = group_terms(H2_STO3G_JW, H2_COEFFS, "FC", budget=5)
    assert r.valid and r.k == 2          # Z-type terms commute; the four XY terms commute with each other
    assert r.k >= r.lb


# T20
@pytest.mark.parametrize("rel", [QWC, FC])
def test_exhaustive_optimum_on_tiny_instances(rel):
    rng = np.random.default_rng(30)
    checked = 0
    while checked < 25:
        m = int(rng.integers(2, 9))
        strings = list(dict.fromkeys(random_strings(rng, m, 3, p_identity=0.3)))
        strings = [s for s in strings if s != "III"]
        if len(strings) < 2:
            continue
        ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)) + 2.0)
        g = ConflictGraph(ps, rel)
        edges = [tuple(e) for e in g.edge_list().tolist()]
        chi = brute_force_chi(ps.m, edges)
        r = run_dtp(ps, rel, budget=0.3, seed=checked)
        assert r.valid and r.k == chi, (strings, rel, r.k, chi)
        checked += 1


# T21 and the G3 replay protocol
@pytest.mark.parametrize("rel", [QWC, FC])
def test_determinism_and_replay(rel):
    rng = np.random.default_rng(31)
    strings = list(dict.fromkeys(random_strings(rng, 260, 9, p_identity=0.45)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    timed = run_dtp(ps, rel, budget=1.0, seed=3)
    replays = [run_dtp(ps, rel, seed=3, iteration_budget=timed.tabu_iterations, ig_budget=timed.ig_steps,
                       polish_steps=timed.polish_steps, trace_memory=(i == 0)) for i in range(3)]
    for rep in replays:
        assert rep.colours == timed.colours
        assert rep.k == timed.k and rep.tabu_iterations == timed.tabu_iterations
        assert rep.rhat == timed.rhat
    assert replays[0].peak_traced_bytes > 0
    assert replays[0].stop_reason in ("replay_budget", "lb_reached")


def test_same_seed_same_output_when_search_completes():
    ps = PauliSet.from_terms(H2_STO3G_JW, H2_COEFFS)
    a = run_dtp(ps, QWC, seed=7, budget=5)
    b = run_dtp(ps, QWC, seed=7, budget=5)
    assert a.colours == b.colours and a.k == b.k


# T22
def test_checker_detects_injected_errors():
    ps = PauliSet.from_terms(H2_STO3G_JW, H2_COEFFS)
    r = run_dtp(ps, QWC, budget=2)
    groups = [list(g) for g in r.groups]
    assert check_grouping(ps.strings, groups, QWC)[0]
    # move an XY term into the Z group -> conflict
    zg = next(i for i, g in enumerate(groups) if len(g) > 1)
    xy = next(i for i, g in enumerate(groups) if len(g) == 1)
    bad = [list(g) for g in groups]
    bad[zg].append(bad[xy].pop())
    bad = [g for g in bad if g]
    ok, msg = check_grouping(ps.strings, bad, QWC)
    assert not ok and "not QWC-compatible" in msg
    assert not check_grouping(ps.strings, groups[1:], QWC)[0]                  # missing terms
    assert not check_grouping(ps.strings, groups + [[groups[0][0]]], QWC)[0]   # repeated term
    assert not check_grouping(ps.strings, groups + [[99]], QWC)[0]             # out of range
    assert not check_grouping(ps.strings, groups + [[]], QWC)[0]               # empty group


@pytest.mark.parametrize("kwargs", [dict(seed_method="random"), dict(clique_stop=False),
                                    dict(gamma="full"), dict(do_polish=False),
                                    dict(tabu_params=TabuParams(fixed_tenure=10)), dict(phase3="tabu"),
                                    dict(phase3="ig"), dict(ig_stall=2, tabu_cap=50)])
def test_ablation_switches_produce_valid_groupings(kwargs):
    rng = np.random.default_rng(32)
    strings = list(dict.fromkeys(random_strings(rng, 150, 8)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    r = run_dtp(ps, QWC, budget=0.5, seed=1, **kwargs)
    assert r.valid and r.k <= r.k_seed and r.k >= r.lb
    if not kwargs.get("do_polish", True):
        assert r.polish_stopped == "off" and r.rhat == pytest.approx(r.rhat_before_polish)
    else:
        assert r.rhat >= r.rhat_before_polish - 1e-12


def test_invalid_options_rejected():
    ps = PauliSet.from_terms(H2_STO3G_JW, H2_COEFFS)
    with pytest.raises(ValueError):
        run_dtp(ps, QWC, seed_method="lf")
    with pytest.raises(ValueError):
        run_dtp(ps, QWC, phase3="annealing")


@pytest.mark.parametrize("phase3", ["hybrid", "tabu", "ig"])
def test_replay_for_every_phase3_mode(phase3):
    rng = np.random.default_rng(34)
    strings = list(dict.fromkeys(random_strings(rng, 220, 8, p_identity=0.4)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    timed = run_dtp(ps, FC, budget=0.8, seed=5, phase3=phase3, ig_stall=5, tabu_cap=300)
    rep = run_dtp(ps, FC, seed=5, phase3=phase3, ig_stall=5, tabu_cap=300, ig_budget=timed.ig_steps,
                  iteration_budget=timed.tabu_iterations, polish_steps=timed.polish_steps)
    assert rep.colours == timed.colours and rep.k_after_reduce == timed.k_after_reduce
    assert rep.ig_steps == timed.ig_steps and rep.tabu_iterations == timed.tabu_iterations
    if phase3 == "tabu":
        assert timed.ig_steps == 0
    if phase3 == "ig":
        assert timed.tabu_iterations == 0 and timed.rounds == 0


def test_empty_and_original_index_mapping():
    r = group_terms([], [], "QWC")
    assert r.k == 0 and r.valid
    r = group_terms(["XI", "XI", "IZ", "II"], [1.0, 1.0, 2.0, 5.0], "QWC", budget=1)
    assert r.k == 1
    assert sorted(i for g in r.original_groups for i in g) == [0, 1, 2]
    s = r.summary()
    assert "groups" not in s and s["k"] == 1 and "timings" in s and "cpu_timings" in s


def test_packed_graph_mode_runs():
    rng = np.random.default_rng(33)
    strings = list(dict.fromkeys(random_strings(rng, 120, 7)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    for mode in ("packed", "implicit"):
        r = run_dtp(ps, FC, budget=0.3, graph_mode=mode)
        assert r.valid and r.graph_mode == mode
