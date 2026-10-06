"""Independent validity checker (gate G1/G2 of the pre-registered protocol).

The checker deliberately does NOT use the bit encoding or the conflict graph: it re-derives
compatibility from the Pauli letters, so an error in the encoding cannot validate itself.
Original code written for this project.
"""
from __future__ import annotations

from typing import Sequence

from .pauli import letters_compatible


def check_grouping(strings: Sequence[str], groups: Sequence[Sequence[int]], relation: str) -> tuple[bool, str]:
    """Return (ok, message). ok is True iff every index 0..m-1 appears in exactly one group
    and every pair of terms inside each group is compatible under ``relation``."""
    m = len(strings)
    seen = [0] * m
    for g in groups:
        for i in g:
            if not (0 <= i < m):
                return False, f"index {i} out of range"
            seen[i] += 1
    missing = [i for i, s in enumerate(seen) if s == 0]
    repeated = [i for i, s in enumerate(seen) if s > 1]
    if missing:
        return False, f"{len(missing)} term(s) not assigned, e.g. {missing[:5]}"
    if repeated:
        return False, f"{len(repeated)} term(s) assigned more than once, e.g. {repeated[:5]}"
    for gi, g in enumerate(groups):
        if len(g) == 0:
            return False, f"group {gi} is empty"
        members = list(g)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                p, q = strings[members[a]], strings[members[b]]
                if not letters_compatible(p, q, relation):
                    return False, f"group {gi}: {p} and {q} are not {relation}-compatible"
    return True, "valid"


def groups_from_colours(col) -> list[list[int]]:
    """Convert a colour array into a list of groups (lists of term indices)."""
    if len(col) == 0:
        return []
    k = int(max(col)) + 1
    groups: list[list[int]] = [[] for _ in range(k)]
    for i, c in enumerate(col):
        groups[int(c)].append(i)
    return [g for g in groups if g]
