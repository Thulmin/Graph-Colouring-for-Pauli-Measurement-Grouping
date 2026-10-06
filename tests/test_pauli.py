"""Phase 0 tests: validation, encoding and conflict tests (T01-T11, T26)."""
from __future__ import annotations

import numpy as np
import pytest

from dtp import FC, QWC, ConflictGraph, PauliSet, letters_compatible, validate_terms
from dtp.pauli import conflict_block, decode_bits, encode_strings, popcount_u64, realise_graph

from conftest import random_strings

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def matrix(s: str) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for ch in s:
        out = np.kron(out, PAULI[ch])
    return out


# T01
def test_reject_wrong_alphabet():
    with pytest.raises(ValueError, match="XQ"):
        validate_terms(["XQ"], [1.0])


# T02
def test_reject_unequal_lengths():
    with pytest.raises(ValueError, match="length"):
        validate_terms(["XI", "Z"], [1.0, 1.0])


# T03
@pytest.mark.parametrize("coeff", [float("nan"), float("inf"), 1j, 0.5 + 0.1j])
def test_reject_non_finite_or_complex(coeff):
    with pytest.raises(ValueError):
        validate_terms(["XI"], [coeff])


def test_reject_non_string_and_empty_and_count_mismatch():
    with pytest.raises(ValueError):
        validate_terms([3], [1.0])
    with pytest.raises(ValueError):
        validate_terms([""], [1.0])
    with pytest.raises(ValueError, match="coefficients"):
        validate_terms(["XI", "IZ"], [1.0])


def test_lower_case_accepted_and_default_coefficients():
    s, a = validate_terms(["xz"])
    assert s == ["XZ"] and a.tolist() == [1.0]


# T04
def test_empty_input():
    ps = PauliSet.from_terms([], [])
    assert ps.m == 0


# T05
def test_identity_only_removed():
    ps = PauliSet.from_terms(["II", "II"], [0.5, 0.25])
    assert ps.m == 0 and ps.identity_coeff == pytest.approx(0.75)


# T06
def test_duplicates_merged_and_small_dropped():
    ps = PauliSet.from_terms(["XZ", "XZ", "ZZ", "IZ"], [1.0, 2.0, 1e-12, 0.5])
    assert ps.strings == ["XZ", "IZ"]
    assert ps.coeffs.tolist() == [3.0, 0.5]
    assert ps.merged_duplicates == 1 and ps.dropped_small == 1
    assert ps.original_indices == [[0, 1], [3]]


# T09
def test_encode_decode_round_trip():
    rng = np.random.default_rng(0)
    for n in (1, 5, 63, 64, 65, 130):
        strings = random_strings(rng, 50, n)
        x, z = encode_strings(strings)
        assert x.shape == (50, -(-n // 64))
        assert decode_bits(x, z, n) == strings


def test_popcount_matches_python():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2 ** 63, size=1000, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, 1000).astype(np.uint64)
    expected = np.array([bin(int(v)).count("1") for v in a])
    assert np.array_equal(popcount_u64(a), expected)


# T10
def test_qwc_bit_test_equals_letter_definition():
    rng = np.random.default_rng(2)
    total = 0
    for n in (3, 17, 64, 65):
        strings = random_strings(rng, 160, n, p_identity=0.6)
        x, z = encode_strings(strings)
        blk = conflict_block(x, z, x, z, QWC)
        for i in range(len(strings)):
            for j in range(len(strings)):
                assert blk[i, j] == (not letters_compatible(strings[i], strings[j], QWC))
                total += 1
    assert total >= 10_000


# T11
def test_fc_bit_test_equals_matrix_commutator():
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 4):
        strings = random_strings(rng, 40, n, p_identity=0.3)
        x, z = encode_strings(strings)
        blk = conflict_block(x, z, x, z, FC)
        mats = [matrix(s) for s in strings]
        for i in range(len(strings)):
            for j in range(len(strings)):
                commute = np.allclose(mats[i] @ mats[j], mats[j] @ mats[i])
                assert blk[i, j] == (not commute)
                assert letters_compatible(strings[i], strings[j], FC) == commute


def test_fc_bit_test_large_n_matches_letters():
    rng = np.random.default_rng(4)
    strings = random_strings(rng, 120, 100, p_identity=0.5)
    x, z = encode_strings(strings)
    blk = conflict_block(x, z, x, z, FC)
    for i in range(0, 120, 7):
        for j in range(120):
            assert blk[i, j] == (not letters_compatible(strings[i], strings[j], FC))


def test_relation_names_checked():
    with pytest.raises(ValueError):
        letters_compatible("X", "Z", "anticommuting")
    with pytest.raises(ValueError):
        letters_compatible("XX", "Z", QWC)


def test_realise_graph_lemma3():
    edges = [(0, 1), (1, 2), (3, 4)]
    strings = realise_graph(6, edges)          # vertex 5 is isolated
    assert len(set(strings)) == 6
    for rel in (QWC, FC):
        ps = PauliSet.from_terms(strings, [1.0] * 6)
        g = ConflictGraph(ps, rel)
        got = {(int(u), int(v)) for u, v in g.edge_list()}
        assert got == set(edges)
    with pytest.raises(ValueError):
        realise_graph(3, [(0, 0)])


# T26
@pytest.mark.parametrize("rel", [QWC, FC])
def test_dense_packed_implicit_equivalence(rel):
    rng = np.random.default_rng(5)
    strings = list(dict.fromkeys(random_strings(rng, 2100, 14, p_identity=0.5)))
    ps = PauliSet.from_terms(strings, rng.normal(size=len(strings)))
    dense = ConflictGraph(ps, rel, mode="dense")
    packed = ConflictGraph(ps, rel, mode="packed")
    implicit = ConflictGraph(ps, rel, mode="implicit")
    assert np.array_equal(dense.degrees, packed.degrees)
    assert np.array_equal(dense.degrees, implicit.degrees)
    for i in rng.choice(ps.m, size=60, replace=False):
        assert np.array_equal(dense.row(i), packed.row(i))
        assert np.array_equal(dense.row(i), implicit.row(i))
    assert implicit.nbytes == 0 and packed.nbytes < dense.nbytes
    assert dense.num_edges == packed.num_edges == implicit.num_edges


def test_conflict_graph_helpers():
    ps = PauliSet.from_terms(["XI", "ZI", "IZ"], [1, 2, 3])
    g = ConflictGraph(ps, QWC)
    assert g.conflicts(0, 1) and not g.conflicts(0, 2)
    assert g.count_monochromatic(np.array([0, 0, 0])) == 1
    assert g.density == pytest.approx(1 / 3)
    assert g.dense_matrix().shape == (3, 3)
    with pytest.raises(ValueError):
        ConflictGraph(ps, QWC, mode="sparse")
