"""Outcome metrics: Crawford et al.'s R-hat, the shot proxy S and QWC measurement bases.

R-hat (Crawford et al., 2021, Quantum 5, 385, Eq. 20):
    R_hat = ( sum_i sum_j |a_ij| / sum_i sqrt( sum_j |a_ij|^2 ) )^2
where i runs over groups and j over the terms of group i. Higher is better; it is the
estimated reduction in shots relative to measuring every term separately under the paper's
state-independent averaging assumption.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def rhat(groups: Sequence[Sequence[int]], coeffs: np.ndarray) -> float:
    """Crawford et al. (2021) R-hat of a grouping (Eq. 20); NaN for an empty grouping."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if len(groups) == 0:
        return float("nan")
    num = float(np.abs(coeffs[[i for g in groups for i in g]]).sum())
    den = float(sum(np.sqrt(np.sum(coeffs[list(g)] ** 2)) for g in groups))
    return (num / den) ** 2 if den > 0 else float("nan")


def qwc_basis(strings: Sequence[str]) -> str:
    """Single-qubit measurement setting for a QWC group: the shared non-identity letter per
    qubit, or '*' (any) where every member acts as identity. Raises if not QWC."""
    n = len(strings[0])
    out = []
    for q in range(n):
        letters = {s[q] for s in strings if s[q] != "I"}
        if len(letters) > 1:
            raise ValueError(f"group is not qubit-wise commuting at qubit {q}: {sorted(letters)}")
        out.append(letters.pop() if letters else "*")
    return "".join(out)
