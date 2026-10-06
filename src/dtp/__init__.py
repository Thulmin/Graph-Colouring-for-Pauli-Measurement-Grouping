"""dtp: DSATUR-Tabu-Polish measurement grouping for Pauli-sum Hamiltonians.

Public API:
    PauliSet.from_terms(terms, coeffs)  - validate and encode Pauli terms (Phase 0)
    run_dtp(pset, relation, ...)        - run the proposed method
    group_terms(terms, coeffs, ...)     - convenience wrapper
    check_grouping(strings, groups, relation) - independent validity checker
"""
from .pauli import PauliSet, QWC, FC, letters_compatible, validate_terms
from .conflicts import ConflictGraph
from .dtp import run_dtp, group_terms, default_budget, DTPResult
from .checker import check_grouping, groups_from_colours
from .metrics import rhat, qwc_basis

__all__ = ["PauliSet", "QWC", "FC", "letters_compatible", "validate_terms", "ConflictGraph",
           "run_dtp", "group_terms", "default_budget", "DTPResult", "check_grouping",
           "groups_from_colours", "rhat", "qwc_basis"]
__version__ = "1.0.0"
