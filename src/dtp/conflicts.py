"""Conflict-graph representations for Pauli measurement grouping (design dossier, section 8).

Three interchangeable representations expose the same interface (``row(i)``, ``degrees``):

* ``dense``    : boolean matrix (1 byte per pair), fastest row access, m^2 bytes.
* ``packed``   : bit-packed rows (np.packbits), m^2/8 bytes, rows unpacked on demand.
* ``implicit`` : no stored adjacency; rows recomputed from the Pauli bit encoding (O(m) memory).

The default ``auto`` mode chooses dense for m <= 20,000, packed for m <= 100,000 and
implicit beyond that. Original code written for this project.
"""
from __future__ import annotations

import numpy as np

from .pauli import PauliSet, conflict_block, _check_relation

DENSE_LIMIT = 20_000
PACKED_LIMIT = 100_000


class ConflictGraph:
    """Conflict graph G=(V,E) of a :class:`PauliSet` under a compatibility relation."""

    def __init__(self, pset: PauliSet, relation: str, mode: str = "auto", block: int = 256):
        self.relation = _check_relation(relation)
        self.pset = pset
        self.m = pset.m
        if mode == "auto":
            mode = "dense" if self.m <= DENSE_LIMIT else ("packed" if self.m <= PACKED_LIMIT else "implicit")
        if mode not in ("dense", "packed", "implicit"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self._dense = None
        self._packed = None
        self._block = block
        deg = np.zeros(self.m, dtype=np.int64)
        if mode == "dense":
            self._dense = np.zeros((self.m, self.m), dtype=bool)
        elif mode == "packed":
            self._packed = np.zeros((self.m, (self.m + 7) // 8), dtype=np.uint8)
        for start in range(0, self.m, block):
            stop = min(self.m, start + block)
            blk = conflict_block(pset.x[start:stop], pset.z[start:stop], pset.x, pset.z, self.relation)
            deg[start:stop] = blk.sum(axis=1)
            if mode == "dense":
                self._dense[start:stop] = blk
            elif mode == "packed":
                self._packed[start:stop] = np.packbits(blk, axis=1)
        self.degrees = deg
        self.num_edges = int(deg.sum() // 2)

    # ------------------------------------------------------------------ access
    def row(self, i: int) -> np.ndarray:
        """Boolean array of length m: True where term j conflicts with term i."""
        if self.mode == "dense":
            return self._dense[i]
        if self.mode == "packed":
            return np.unpackbits(self._packed[i], count=self.m).astype(bool)
        return self.pset.conflict_row(i, self.relation)

    def neighbours(self, i: int) -> np.ndarray:
        """Indices of the conflict neighbours of term i."""
        return np.flatnonzero(self.row(i))

    def conflicts(self, i: int, j: int) -> bool:
        """True if terms i and j are not compatible."""
        return bool(self.row(i)[j])

    @property
    def density(self) -> float:
        """Edge density 2|E| / (m(m-1)) of the conflict graph."""
        return 0.0 if self.m < 2 else 2.0 * self.num_edges / (self.m * (self.m - 1))

    @property
    def nbytes(self) -> int:
        """Bytes used by the stored adjacency (0 for the implicit representation)."""
        if self.mode == "dense":
            return int(self._dense.nbytes)
        if self.mode == "packed":
            return int(self._packed.nbytes)
        return 0

    def dense_matrix(self) -> np.ndarray:
        """Return a dense boolean matrix (materialised if necessary; small graphs only)."""
        if self.mode == "dense":
            return self._dense
        return np.vstack([self.row(i) for i in range(self.m)]) if self.m else np.zeros((0, 0), bool)

    # ------------------------------------------------------------------ helpers
    def count_monochromatic(self, colours: np.ndarray) -> int:
        """Number of conflict edges whose endpoints share a colour (0 means proper)."""
        total = 0
        for i in range(self.m):
            r = self.row(i)
            total += int(np.count_nonzero(r & (colours == colours[i])))
        return total // 2

    def edge_list(self) -> np.ndarray:
        """Edges (i<j) as an array of shape (|E|, 2); intended for baselines on small graphs."""
        out = []
        for i in range(self.m):
            js = np.flatnonzero(self.row(i))
            js = js[js > i]
            if js.size:
                out.append(np.column_stack([np.full(js.size, i), js]))
        return np.vstack(out) if out else np.zeros((0, 2), dtype=np.int64)
