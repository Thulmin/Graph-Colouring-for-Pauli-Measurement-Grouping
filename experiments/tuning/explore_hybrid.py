# Exploration script run on the TUNING molecules (HF, BH3) on 1 October 2026, before any
# evaluation run. Kept verbatim as the record of the Phase 3 amendment (docs/DEVIATIONS.md).
# Results: results/tuning/class_choice.csv and results/tuning/hybrid_phase3.csv.
"""Tuning-set exploration: IG + TabuCol alternation for Phase 3 (HF, BH3 only)."""
import sys, time, math, json, numpy as np
sys.path.insert(0, '/home/claude/project/pauli-grouping-dtp/src')
from dtp import PauliSet, ConflictGraph
from dtp.instances import load
from dtp.colouring import dsatur, greedy_clique, num_colours, renumber, greedy_first_fit
from dtp.dtp import default_budget
from dtp.tabucol import reduce_one, TabuParams

def ig_step(g, col, k, rng):
    rule = int(rng.integers(3))
    if rule == 0: order = np.arange(k - 1, -1, -1)
    elif rule == 1: order = np.argsort(-np.bincount(col, minlength=k), kind='stable')
    else: order = rng.permutation(k)
    rank = np.empty(k, np.int64); rank[order] = np.arange(k)
    perm = np.argsort(rank[col], kind='stable')
    return renumber(greedy_first_fit(g, perm))

def hybrid(name, rel, seed, stall, cap, backoff=0):
    inst = load(name); ps = PauliSet.from_terms(inst.terms, inst.coeffs); g = ConflictGraph(ps, rel)
    col = dsatur(g); k = num_colours(col); lb, _ = greedy_clique(g)
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); deadline = t0 + default_budget(ps.m)
    params = TabuParams(tenure_L=50)
    ig_its = tabu_rounds = tabu_succ = 0; since = 0; cur = stall
    while k > lb and time.perf_counter() < deadline:
        if since < cur:
            col = ig_step(g, col, k, rng); ig_its += 1
            k2 = num_colours(col)
            if k2 < k: k = k2; since = 0
            else: since += 1
        else:
            ok, h, its = reduce_one(g, col, k, deadline, rng, params, max_iter=cap); tabu_rounds += 1
            if ok: col, k = renumber(h), k - 1; tabu_succ += 1; cur = stall
            elif backoff: cur = cur * 2
            since = 0
    return k, ig_its, tabu_rounds, tabu_succ

if __name__ == '__main__':
    name, rel, seed, stall, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    bo = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    print(json.dumps([name, rel, seed, stall, cap, bo, *hybrid(name, rel, seed, stall, cap, bo)]), flush=True)
