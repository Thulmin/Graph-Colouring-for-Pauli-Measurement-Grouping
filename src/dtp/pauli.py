"""Pauli-string validation, symplectic bit encoding and conflict tests.

This module implements Phase 0 (ENCODE) of the DSATUR-Tabu-Polish (DTP) design and
Lemma 1 of the design dossier (bitwise conflict tests).

Encoding (per qubit q):  I=(x=0,z=0)  X=(1,0)  Z=(0,1)  Y=(1,1).
A Pauli string P of length n is stored as two bit vectors x, z of n bits, packed into
W = ceil(n/64) unsigned 64-bit words.  Qubit q is bit (q mod 64) of word (q // 64).
String index 0 is qubit 0 (little-endian with respect to the string), which is the
OpenFermion convention used by our instance generator.

Original code written for this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

QWC = "QWC"  # qubit-wise commutativity
FC = "FC"    # full (general) commutativity
RELATIONS = (QWC, FC)

_ALPHABET = frozenset("IXYZ")
_X_BIT = {"I": 0, "X": 1, "Y": 1, "Z": 0}
_Z_BIT = {"I": 0, "X": 0, "Y": 1, "Z": 1}


def _check_relation(relation: str) -> str:
    rel = str(relation).upper()
    if rel not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}, got {relation!r}")
    return rel


def popcount_u64(a: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64 element (vectorised).

    Uses ``np.bitwise_count`` (NumPy >= 2.0) when available, otherwise a SWAR fallback.
    """
    a = np.asarray(a, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).astype(np.int64)
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def letters_compatible(p: str, q: str, relation: str) -> bool:
    """Reference (letter-level) compatibility test, independent of the bit encoding.

    QWC: on every qubit the letters are equal or at least one is 'I'.
    FC : the number of qubits where both letters are non-identity and different is even.
    Used by the independent checker and by property tests of the bit tests.
    """
    rel = _check_relation(relation)
    if len(p) != len(q):
        raise ValueError("Pauli strings must have equal length")
    clashes = sum(1 for a, b in zip(p, q) if a != "I" and b != "I" and a != b)
    return clashes == 0 if rel == QWC else clashes % 2 == 0


def realise_graph(num_vertices: int, edges: Iterable[tuple[int, int]]) -> list[str]:
    """Pauli strings whose QWC and FC conflict graphs both equal the given simple graph.

    Construction of Lemma 3 (design dossier): one qubit per edge e = (u, v), with X on u and Z on
    v at that qubit and I elsewhere; each isolated vertex gets one extra qubit carrying Z. Two
    strings then share a non-identity qubit only if they are adjacent, and an adjacent pair
    differs on exactly one such qubit (X versus Z), which is a conflict under QWC and FC.
    """
    edge_list = []
    seen = set()
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v or not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"invalid edge ({u}, {v})")
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            edge_list.append(key)
    deg = [0] * num_vertices
    for u, v in edge_list:
        deg[u] += 1
        deg[v] += 1
    isolated = [v for v in range(num_vertices) if deg[v] == 0]
    n = len(edge_list) + len(isolated)
    rows = [["I"] * max(n, 1) for _ in range(num_vertices)]
    for q, (u, v) in enumerate(edge_list):
        rows[u][q] = "X"
        rows[v][q] = "Z"
    for t, v in enumerate(isolated):
        rows[v][len(edge_list) + t] = "Z"
    return ["".join(r) for r in rows]


def validate_terms(terms: Sequence[str], coeffs: Sequence[complex] | None = None,
                   imag_tol: float = 1e-12) -> tuple[list[str], np.ndarray]:
    """Validate Pauli strings and coefficients; return (strings, real coefficients).

    Raises ValueError with a message naming the offending term when:
      * a term is not a string, is empty, or contains letters outside {I,X,Y,Z};
      * terms have different lengths;
      * a coefficient is not finite or has an imaginary part larger than ``imag_tol``.
    Lower-case letters are accepted and upper-cased.
    """
    terms = list(terms)
    if coeffs is None:
        coeffs = [1.0] * len(terms)
    coeffs = list(coeffs)
    if len(coeffs) != len(terms):
        raise ValueError(f"{len(terms)} terms but {len(coeffs)} coefficients")
    if not terms:
        return [], np.zeros(0, dtype=np.float64)
    clean: list[str] = []
    for i, t in enumerate(terms):
        if not isinstance(t, str) or len(t) == 0:
            raise ValueError(f"term {i} must be a non-empty string, got {t!r}")
        u = t.upper()
        bad = set(u) - _ALPHABET
        if bad:
            raise ValueError(f"term {i} ({t!r}) contains letters outside I,X,Y,Z: {sorted(bad)}")
        clean.append(u)
    n = len(clean[0])
    for i, t in enumerate(clean):
        if len(t) != n:
            raise ValueError(f"term {i} has length {len(t)}, expected {n} (all terms must act on the same qubits)")
    out = np.empty(len(coeffs), dtype=np.float64)
    for i, c in enumerate(coeffs):
        c = complex(c)
        if not (np.isfinite(c.real) and np.isfinite(c.imag)):
            raise ValueError(f"coefficient {i} is not finite: {c!r}")
        if abs(c.imag) > imag_tol:
            raise ValueError(f"coefficient {i} has a non-negligible imaginary part {c.imag!r}; "
                             "a Hermitian Hamiltonian in the Pauli basis has real coefficients")
        out[i] = c.real
    return clean, out


def encode_strings(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode equal-length Pauli strings as (x, z) arrays of shape (m, W) and dtype uint64."""
    m = len(strings)
    n = len(strings[0]) if m else 0
    W = max(1, -(-n // 64))
    x = np.zeros((m, W), dtype=np.uint64)
    z = np.zeros((m, W), dtype=np.uint64)
    for i, s in enumerate(strings):
        xw = [0] * W
        zw = [0] * W
        for q, ch in enumerate(s):
            if ch == "I":
                continue
            w, b = divmod(q, 64)
            if _X_BIT[ch]:
                xw[w] |= 1 << b
            if _Z_BIT[ch]:
                zw[w] |= 1 << b
        x[i] = xw
        z[i] = zw
    return x, z


def decode_bits(x: np.ndarray, z: np.ndarray, n: int) -> list[str]:
    """Inverse of :func:`encode_strings` (used by the round-trip test)."""
    out = []
    lut = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    for xi, zi in zip(x, z):
        chars = []
        for q in range(n):
            w, b = divmod(q, 64)
            chars.append(lut[(int(xi[w] >> np.uint64(b)) & 1, int(zi[w] >> np.uint64(b)) & 1)])
        out.append("".join(chars))
    return out


def conflict_block(xb: np.ndarray, zb: np.ndarray, x: np.ndarray, z: np.ndarray,
                   relation: str) -> np.ndarray:
    """Conflict matrix between a block of terms (b, W) and all terms (m, W).

    Returns a boolean array of shape (b, m); entry [r, j] is True when term r of the
    block and term j are NOT compatible under ``relation`` (Lemma 1 of the dossier).
    The diagonal (a term against itself) is always compatible, hence False.
    """
    rel = _check_relation(relation)
    xb = xb[:, None, :]
    zb = zb[:, None, :]
    xa = x[None, :, :]
    za = z[None, :, :]
    if rel == QWC:
        support = (xb | zb) & (xa | za)
        differ = (xb ^ xa) | (zb ^ za)
        return ((support & differ) != 0).any(axis=2)
    sym = (xb & za) ^ (zb & xa)
    parity = popcount_u64(sym).sum(axis=2) & 1
    return parity == 1


@dataclass
class PauliSet:
    """Cleaned, deduplicated non-identity Pauli terms of a Hamiltonian (Phase 0 output)."""

    n: int
    strings: list[str]
    coeffs: np.ndarray
    x: np.ndarray
    z: np.ndarray
    identity_coeff: float = 0.0
    original_indices: list[list[int]] = field(default_factory=list)
    dropped_small: int = 0
    merged_duplicates: int = 0

    @property
    def m(self) -> int:
        """Number of cleaned non-identity terms."""
        return len(self.strings)

    @classmethod
    def from_terms(cls, terms: Sequence[str], coeffs: Sequence[complex] | None = None,
                   threshold: float = 1e-8) -> "PauliSet":
        """Validate, remove the identity, merge duplicates and drop |a| < threshold."""
        strings, a = validate_terms(terms, coeffs)
        if not strings:
            return cls(0, [], np.zeros(0), np.zeros((0, 1), np.uint64), np.zeros((0, 1), np.uint64))
        n = len(strings[0])
        ident = "I" * n
        index: dict[str, int] = {}
        kept: list[str] = []
        acc: list[float] = []
        origin: list[list[int]] = []
        identity_coeff = 0.0
        merged = 0
        for i, (s, c) in enumerate(zip(strings, a)):
            if s == ident:
                identity_coeff += c
                continue
            j = index.get(s)
            if j is None:
                index[s] = len(kept)
                kept.append(s)
                acc.append(c)
                origin.append([i])
            else:
                acc[j] += c
                origin[j].append(i)
                merged += 1
        acc_arr = np.asarray(acc, dtype=np.float64)
        keep = np.abs(acc_arr) >= threshold
        dropped = int((~keep).sum())
        kept = [s for s, k in zip(kept, keep) if k]
        origin = [o for o, k in zip(origin, keep) if k]
        acc_arr = acc_arr[keep]
        x, z = encode_strings(kept) if kept else (np.zeros((0, 1), np.uint64), np.zeros((0, 1), np.uint64))
        return cls(n, kept, acc_arr, x, z, identity_coeff, origin, dropped, merged)

    def conflict_row(self, i: int, relation: str) -> np.ndarray:
        """Boolean conflict row of term i against all terms (implicit adjacency)."""
        return conflict_block(self.x[i:i + 1], self.z[i:i + 1], self.x, self.z, relation)[0]
