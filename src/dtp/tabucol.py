"""Phase 3 of DTP: k-reduction with TabuCol using lazy conflict-count (gamma) rows.

TabuCol is due to Hertz & de Werra (1987); the dynamic tenure rule
t = rand(0..L-1) + floor(lambda * |CV|) follows Galinier & Hao (1999).

Data-structure adaptation (design dossier, section 8). Classic TabuCol stores the full
m x k matrix gamma[v][c] = number of neighbours of v with colour c. When k is about m/3, as for
QWC Pauli grouping, that matrix needs O(m^2) memory. Here gamma rows exist only for the
*currently conflicting* vertices CV ("lazy" rows): a row is computed from the conflict row when
a vertex enters CV, is updated in O(1) when one of its neighbours moves, and is released when
the vertex leaves CV. Memory is O(m + |CV| k) instead of O(m k).

Three implementations with identical search semantics are provided:

* ``tabucol(..., gamma="lazy")``  - the DTP implementation (lazy cached rows, vectorised);
* ``tabucol(..., gamma="full")``  - classic full gamma matrix (ablation A3);
* :func:`tabucol_reference`        - a direct, unoptimised transcription of the pseudocode,
  kept for differential testing (same RNG calls, same tie-breaking, same tabu rules).

Given the same input, seed and iteration count, all three return the same colouring.
Original implementation written for this project.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .conflicts import ConflictGraph

_BIG = np.iinfo(np.int64).max // 4
_PURGE_EVERY = 2000


@dataclass
class TabuParams:
    """TabuCol parameters (defaults: Galinier & Hao's dynamic tenure rule, L = 10, lambda = 0.6)."""

    tenure_L: int = 10          # random part drawn from 0..L-1
    tenure_lambda: float = 0.6  # proportional part (times number of conflicting vertices)
    fixed_tenure: int | None = None  # ablation A5: constant tenure instead of the dynamic rule
    max_iter_per_round: int | None = None


def _initial_same_colour_counts(graph: ConflictGraph, col: np.ndarray) -> np.ndarray:
    """cc[v] = number of neighbours of v that share v's colour (O(m^2))."""
    cc = np.zeros(graph.m, dtype=np.int64)
    for v in range(graph.m):
        cc[v] = np.count_nonzero(graph.row(v) & (col == col[v]))
    return cc


def _tenure(params: TabuParams, rng: np.random.Generator, n_conflicting: int) -> int:
    if params.fixed_tenure is not None:
        return int(params.fixed_tenure)
    return int(rng.integers(0, params.tenure_L)) + int(params.tenure_lambda * n_conflicting)


class _TabuList:
    """Tabu entries (v, c) -> expiry iteration, stored in arrays for vectorised masking.

    A move of v to colour c is tabu at iteration ``it`` while expiry > it. Re-adding (v, c)
    overwrites the expiry (same semantics as a dict). Expired entries are purged periodically.
    """

    def __init__(self, m: int, k: int):
        self.k = k
        self.index: dict[int, int] = {}
        self.TV = np.empty(64, dtype=np.int64)
        self.TC = np.empty(64, dtype=np.int64)
        self.TE = np.empty(64, dtype=np.int64)
        self.n = 0
        self.pos = np.full(m, -1, dtype=np.int64)

    def add(self, v: int, c: int, expiry: int) -> None:
        key = v * self.k + c
        p = self.index.get(key)
        if p is None:
            if self.n == self.TV.size:
                for name in ("TV", "TC", "TE"):
                    arr = getattr(self, name)
                    new = np.empty(arr.size * 2, dtype=np.int64)
                    new[:arr.size] = arr
                    setattr(self, name, new)
            p = self.n
            self.n += 1
            self.index[key] = p
            self.TV[p] = v
            self.TC[p] = c
        self.TE[p] = expiry

    def mask(self, D: np.ndarray, CV: np.ndarray, it: int, f: int, best_f: int) -> None:
        """Set D[row, c] = BIG for tabu moves that do not satisfy the aspiration criterion."""
        if self.n == 0:
            return
        act = np.flatnonzero(self.TE[:self.n] > it)
        if act.size == 0:
            return
        self.pos[CV] = np.arange(CV.size)
        rows = self.pos[self.TV[act]]
        self.pos[CV] = -1
        sel = rows >= 0
        if not sel.any():
            return
        r = rows[sel]
        c = self.TC[act[sel]]
        blocked = f + D[r, c] >= best_f
        D[r[blocked], c[blocked]] = _BIG

    def purge(self, it: int) -> None:
        keep = np.flatnonzero(self.TE[:self.n] > it)
        tv, tc, te = self.TV[keep].copy(), self.TC[keep].copy(), self.TE[keep].copy()
        self.n = keep.size
        self.TV[:self.n], self.TC[:self.n], self.TE[:self.n] = tv, tc, te
        k = self.k
        self.index = {int(v) * k + int(c): i for i, (v, c) in enumerate(zip(tv.tolist(), tc.tolist()))}


class _LazyGamma:
    """Gamma rows for the conflicting vertices only (cached; O(|CV| k) memory)."""

    def __init__(self, graph: ConflictGraph, k: int):
        self.graph = graph
        self.k = k
        self.slot = np.full(graph.m, -1, dtype=np.int64)
        cap = 32
        self.rows = np.zeros((cap, k), dtype=np.int64)
        self.owner = np.full(cap, -1, dtype=np.int64)
        self.free = list(range(cap - 1, -1, -1))
        self.peak_rows = 0

    def _grow(self) -> None:
        cap = self.owner.size
        rows = np.zeros((2 * cap, self.k), dtype=np.int64)
        rows[:cap] = self.rows
        owner = np.full(2 * cap, -1, dtype=np.int64)
        owner[:cap] = self.owner
        self.rows, self.owner = rows, owner
        self.free.extend(range(2 * cap - 1, cap - 1, -1))

    def rows_for(self, CV: np.ndarray, incv: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Return the gamma rows of the vertices in CV (in CV order); compute missing rows."""
        used = np.flatnonzero(self.owner >= 0)
        if used.size:
            gone = used[~incv[self.owner[used]]]
            if gone.size:
                self.slot[self.owner[gone]] = -1
                self.owner[gone] = -1
                self.free.extend(gone.tolist())
        s = self.slot[CV]
        missing = np.flatnonzero(s < 0)
        for i in missing.tolist():
            v = int(CV[i])
            if not self.free:
                self._grow()
            p = self.free.pop()
            self.rows[p] = np.bincount(col[self.graph.row(v)], minlength=self.k)
            self.owner[p] = v
            self.slot[v] = p
            s[i] = p
        n_used = int(np.count_nonzero(self.owner >= 0))
        if n_used > self.peak_rows:
            self.peak_rows = n_used
        return self.rows[s]

    def on_move(self, rv: np.ndarray, old: int, new: int) -> None:
        """Vertex with conflict row ``rv`` moved from colour ``old`` to ``new``."""
        used = np.flatnonzero(self.owner >= 0)
        if used.size == 0:
            return
        hit = used[rv[self.owner[used]]]
        if hit.size:
            self.rows[hit, old] -= 1
            self.rows[hit, new] += 1


def _full_gamma(graph: ConflictGraph, col: np.ndarray, k: int) -> np.ndarray:
    G = np.zeros((graph.m, k), dtype=np.int64)
    for v in range(graph.m):
        G[v] = np.bincount(col[graph.row(v)], minlength=k)
    return G


def tabucol(graph: ConflictGraph, col: np.ndarray, k: int, deadline: float,
            rng: np.random.Generator, params: TabuParams | None = None,
            cc: np.ndarray | None = None, max_iter: int | None = None,
            gamma: str = "lazy", stats: dict | None = None) -> tuple[bool, np.ndarray, int]:
    """Search for a proper k-colouring starting from ``col`` (colours in 0..k-1).

    Each iteration moves one conflicting vertex to the colour that most reduces
    f = number of monochromatic conflict edges. A move that would put v back into a colour it
    left less than ``tenure`` iterations ago is tabu unless it gives f below the best f of this
    call (aspiration). Ties are broken uniformly at random with ``rng``. Stops when f = 0, at
    ``deadline`` (``time.perf_counter`` value) or after ``max_iter`` iterations.

    Returns (success, colours, iterations); ``success`` is True iff f reached 0.
    ``gamma`` selects the lazy rows ("lazy", DTP) or the classic full matrix ("full", A3).
    """
    params = params or TabuParams()
    if gamma not in ("lazy", "full"):
        raise ValueError("gamma must be 'lazy' or 'full'")
    m = graph.m
    col = col.astype(np.int64, copy=True)
    cc = _initial_same_colour_counts(graph, col) if cc is None else cc.astype(np.int64, copy=True)
    f = int(cc.sum() // 2)
    best_f = f
    it = 0
    tabu = _TabuList(m, k)
    incv = np.zeros(m, dtype=bool)
    G = _full_gamma(graph, col, k) if gamma == "full" else None
    lazy = _LazyGamma(graph, k) if gamma == "lazy" else None
    while f > 0:
        if time.perf_counter() >= deadline:
            break
        if max_iter is not None and it >= max_iter:
            break
        if params.max_iter_per_round is not None and it >= params.max_iter_per_round:
            break
        CV = np.flatnonzero(cc > 0)
        if lazy is not None:
            incv[CV] = True
            GV = lazy.rows_for(CV, incv, col)
            incv[CV] = False
        else:
            GV = G[CV]
        ar = np.arange(CV.size)
        cur = col[CV]
        D = GV - GV[ar, cur][:, None]
        D[ar, cur] = _BIG
        tabu.mask(D, CV, it, f, best_f)
        mn = int(D.min())
        if mn >= _BIG:              # every move is tabu: let tenures expire
            it += 1
            continue
        cand = np.flatnonzero(D.ravel() == mn)
        pick = int(cand[int(rng.integers(cand.size))])
        r, c = divmod(pick, k)
        v = int(CV[r])
        old = int(col[v])
        rv = graph.row(v)
        nb = np.flatnonzero(rv)
        nbc = col[nb]
        cc[nb[nbc == old]] -= 1
        cc[nb[nbc == c]] += 1
        cc[v] = int(np.count_nonzero(nbc == c))
        col[v] = c
        if G is not None:
            G[nb, old] -= 1
            G[nb, c] += 1
        else:
            lazy.on_move(rv, old, c)
        f += mn
        tabu.add(v, old, it + _tenure(params, rng, CV.size))
        if f < best_f:
            best_f = f
        it += 1
        if it % _PURGE_EVERY == 0:
            tabu.purge(it)
    if stats is not None:
        stats["gamma_rows_peak"] = (lazy.peak_rows if lazy is not None else m)
        stats["gamma_bytes"] = int(lazy.rows.nbytes if lazy is not None else G.nbytes)
    return f == 0, col, it


def tabucol_reference(graph: ConflictGraph, col: np.ndarray, k: int, deadline: float,
                      rng: np.random.Generator, params: TabuParams | None = None,
                      cc: np.ndarray | None = None, max_iter: int | None = None
                      ) -> tuple[bool, np.ndarray, int]:
    """Unoptimised transcription of the TABUCOL pseudocode (design dossier, section 10).

    Recomputes gamma rows from the conflict rows for every conflicting vertex in every
    iteration. Used only to test that :func:`tabucol` follows exactly the same search path.
    """
    params = params or TabuParams()
    col = col.astype(np.int64, copy=True)
    cc = _initial_same_colour_counts(graph, col) if cc is None else cc.astype(np.int64, copy=True)
    f = int(cc.sum() // 2)
    best_f = f
    it = 0
    tabu: dict[int, dict[int, int]] = {}
    while f > 0:
        if time.perf_counter() >= deadline:
            break
        if max_iter is not None and it >= max_iter:
            break
        if params.max_iter_per_round is not None and it >= params.max_iter_per_round:
            break
        CV = np.flatnonzero(cc > 0)
        best_delta = _BIG
        moves: list[tuple[int, int]] = []
        for v in CV:
            v = int(v)
            gam = np.bincount(col[graph.row(v)], minlength=k)
            cv = col[v]
            delta = gam - gam[cv]
            delta[cv] = _BIG
            for c, expiry in tabu.get(v, {}).items():
                if expiry > it and f + delta[c] >= best_f:   # tabu and no aspiration
                    delta[c] = _BIG
            mn = int(delta.min())
            if mn < best_delta:
                best_delta = mn
                moves = [(v, int(c)) for c in np.flatnonzero(delta == mn)]
            elif mn == best_delta and mn < _BIG:
                moves.extend((v, int(c)) for c in np.flatnonzero(delta == mn))
        if best_delta >= _BIG:
            it += 1
            continue
        v, c = moves[int(rng.integers(len(moves)))]
        old = int(col[v])
        nb = graph.neighbours(v)
        nbc = col[nb]
        cc[nb[nbc == old]] -= 1
        cc[nb[nbc == c]] += 1
        cc[v] = int(np.count_nonzero(nbc == c))
        col[v] = c
        f += best_delta
        tabu.setdefault(v, {})[old] = it + _tenure(params, rng, len(CV))
        if f < best_f:
            best_f = f
        it += 1
        if it % _PURGE_EVERY == 0:
            tabu = {u: {cc_: e for cc_, e in d.items() if e > it} for u, d in tabu.items()}
            tabu = {u: d for u, d in tabu.items() if d}
    return f == 0, col, it


def reduce_one(graph: ConflictGraph, col: np.ndarray, k: int, deadline: float,
               rng: np.random.Generator, params: TabuParams | None = None,
               max_iter: int | None = None, gamma: str = "lazy", stats: dict | None = None
               ) -> tuple[bool, np.ndarray, int]:
    """Try to turn a proper k-colouring into a proper (k-1)-colouring.

    Removes the smallest colour class (ties: lowest colour), renumbers colours, reinserts
    each removed vertex (in index order) into the colour with fewest conflicting neighbours
    (ties: lowest colour), then runs :func:`tabucol` with k-1 colours. Conflict counts are
    initialised incrementally (the input colouring is proper, so only reinserted vertices
    create conflicts). Returns (success, colours, TabuCol iterations).
    """
    m = graph.m
    sizes = np.bincount(col, minlength=k)
    r = int(np.argmin(sizes))
    members = np.flatnonzero(col == r)
    h = col.astype(np.int64, copy=True)
    h[members] = -1
    h[h > r] -= 1
    kk = k - 1
    cc = np.zeros(m, dtype=np.int64)
    for v in members.tolist():
        nb = graph.neighbours(v)
        nbc = h[nb]
        cnt = np.bincount(nbc[nbc >= 0], minlength=kk)
        c = int(np.argmin(cnt))
        h[v] = c
        cc[v] = int(cnt[c])
        cc[nb[nbc == c]] += 1
    return tabucol(graph, h, kk, deadline, rng, params, cc, max_iter=max_iter, gamma=gamma, stats=stats)
