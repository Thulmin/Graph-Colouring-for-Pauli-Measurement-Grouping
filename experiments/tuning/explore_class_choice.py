# Exploration script run on the TUNING molecules (HF, BH3) on 1 October 2026, before any
# evaluation run. Kept verbatim as the record of the Phase 3 amendment (docs/DEVIATIONS.md).
# Results: results/tuning/class_choice.csv and results/tuning/hybrid_phase3.csv.
"""Tuning-set exploration: which colour class should REDUCE-ONE remove? (HF, BH3 only)"""
import sys, time, math, json, numpy as np
sys.path.insert(0, '/home/claude/project/pauli-grouping-dtp/src')
from dtp import PauliSet, ConflictGraph
from dtp.instances import load
from dtp.colouring import dsatur, greedy_clique, num_colours, renumber
from dtp.dtp import default_budget
import dtp.tabucol as T
from dtp import baselines as B

def mincost_per_vertex(g, col, k):
    out = np.empty(g.m, np.int64)
    for v in range(g.m):
        cnt = np.bincount(col[g.row(v)], minlength=k); cnt[col[v]] = 1 << 40; out[v] = cnt.min()
    return out

def reduce_variant(g, col, k, deadline, rng, rule):
    sizes = np.bincount(col, minlength=k)
    if rule == 'smallest':
        r = int(np.argmin(sizes))
    elif rule == 'random':
        r = int(rng.integers(k))
    elif rule == 'mincost':
        mc = mincost_per_vertex(g, col, k)
        cost = np.bincount(col, weights=mc, minlength=k)
        key = cost * (g.m + 1) + sizes
        r = int(np.argmin(key))
    members = np.flatnonzero(col == r)
    h = col.astype(np.int64, copy=True); h[members] = -1; h[h > r] -= 1; kk = k - 1
    cc = np.zeros(g.m, np.int64)
    for v in members.tolist():
        nb = g.neighbours(v); nbc = h[nb]; cnt = np.bincount(nbc[nbc >= 0], minlength=kk)
        c = int(np.argmin(cnt)); h[v] = c; cc[v] = int(cnt[c]); cc[nb[nbc == c]] += 1
    return T.tabucol(g, h, kk, deadline, rng, T.TabuParams(tenure_L=50), cc)

def run(name, rel, rule, seed):
    inst = load(name); ps = PauliSet.from_terms(inst.terms, inst.coeffs); g = ConflictGraph(ps, rel)
    col = dsatur(g); k = num_colours(col); lb, _ = greedy_clique(g)
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); deadline = t0 + default_budget(ps.m)
    rounds = 0
    while k > lb and time.perf_counter() < deadline:
        ok, h, its = reduce_variant(g, col, k, deadline, rng, rule); rounds += 1
        if not ok: break
        col, k = renumber(h), k - 1
    return k, rounds

def run_ig(name, rel, seed):
    inst = load(name); ps = PauliSet.from_terms(inst.terms, inst.coeffs); g = ConflictGraph(ps, rel)
    col = dsatur(g); lb, _ = greedy_clique(g)
    out = B.iterated_greedy(g, col, default_budget(ps.m), seed, lb=lb)
    return num_colours(out.colours), out.extra['iterations']

if __name__ == '__main__':
    name, rel, rule, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if rule == 'IG':
        print(json.dumps([name, rel, rule, seed, *run_ig(name, rel, seed)]), flush=True)
    else:
        print(json.dumps([name, rel, rule, seed, *run(name, rel, rule, seed)]), flush=True)
