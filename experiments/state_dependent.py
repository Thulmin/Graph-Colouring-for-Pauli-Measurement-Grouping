"""Exploratory analysis foreseen in the pre-registration (optional; not one of the hypothesis tests).

For the main molecules with at most 14 qubits, the state-independent shot proxy R-hat is compared with the exact,
state-dependent shot reduction R_psi for the ground state psi of the qubit Hamiltonian in the sector with the
molecule's electron number and S_z = 0 (Jordan-Wigner: qubit q is spin orbital q, even = alpha, odd = beta):

    R_psi(g) = (sum_j |a_j| sigma_j)^2 / (sum_G sigma_G)^2,
    sigma_j^2 = 1 - <P_j>^2,   sigma_G^2 = <A_G^2> - <A_G>^2,   A_G = sum_{j in G} a_j P_j.

R_psi is the ratio of the shots needed to estimate <H> to a fixed precision when every term is measured on its own
to the shots needed with grouping g, both with optimal shot allocation. With all expectation values and all
covariances equal to zero (the maximally mixed state) it reduces to R-hat.

Groupings compared (seed-0 runs of the main phase): DTP after polishing, DTP before polishing (reproduced
exactly by a deterministic replay of the recorded Phase 3 step counts with polishing off), Sorted Insertion,
Qiskit's default and PennyLane's DSATUR option.

Usage: python experiments/state_dependent.py   ->  results/tables/state_dependent.csv
Original code written for this project.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtp import PauliSet, groups_from_colours, rhat, run_dtp  # noqa: E402
from dtp.instances import load  # noqa: E402

INSTANCES = ["H2_sto-3g_JW", "LiH_sto-3g_JW", "BeH2_sto-3g_JW", "H2O_sto-3g_JW"]
METHODS = ["dtp", "dtp_before_polish", "sorted_insertion", "qiskit_lf", "pennylane_dsatur"]


def masks(term: str) -> tuple[int, int, int]:
    """X mask, Z mask and number of Y letters of a Pauli string (character q acts on qubit q = bit q)."""
    xm = zm = 0
    for q, ch in enumerate(term):
        if ch in "XY":
            xm |= 1 << q
        if ch in "ZY":
            zm |= 1 << q
    return xm, zm, term.count("Y")


def apply(psi: np.ndarray, idx: np.ndarray, xm: int, zm: int, ny: int) -> np.ndarray:
    """P|psi> with P|b> = i^ny (-1)^popcount(z & b) |b xor x>."""
    sign = 1 - 2 * (np.bitwise_count(idx & zm).astype(np.int64) & 1)   # bitwise_count is unsigned: cast first
    out = np.empty_like(psi)
    out[idx ^ xm] = (1j ** ny) * sign * psi
    return out


def ground_state(terms: list[str], coeffs: np.ndarray, n: int, n_el: int) -> tuple[float, np.ndarray, int]:
    """Lowest eigenpair of sum_j a_j P_j restricted to popcount = n_el and S_z = 0; the state is embedded in 2^n."""
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    even = sum(1 << q for q in range(0, n, 2))
    odd = sum(1 << q for q in range(1, n, 2))
    n_a = np.bitwise_count(idx & even).astype(np.int64)
    n_b = np.bitwise_count(idx & odd).astype(np.int64)
    sector = idx[(n_a + n_b == n_el) & (n_a == n_b)]
    pos = np.full(dim, -1, dtype=np.int64)
    pos[sector] = np.arange(len(sector))
    h = np.zeros((len(sector), len(sector)), dtype=complex)
    cols = np.arange(len(sector))
    for t, a in zip(terms, coeffs):
        xm, zm, ny = masks(t)
        rows = pos[sector ^ xm]
        keep = rows >= 0
        sign = 1 - 2 * (np.bitwise_count(sector & zm).astype(np.int64) & 1)
        h[rows[keep], cols[keep]] += a * (1j ** ny) * sign[keep]   # one entry per column for a single term
    herm = float(np.abs(h - h.conj().T).max())
    if herm > 1e-9:
        raise RuntimeError(f"sector Hamiltonian not Hermitian ({herm:.2e})")
    w, v = np.linalg.eigh(h)
    psi = np.zeros(dim, dtype=complex)
    psi[sector] = v[:, 0]
    return float(w[0]), psi, len(sector)


def r_psi(groups: list[list[int]], terms: list[str], coeffs: np.ndarray, psi: np.ndarray, sigma: np.ndarray) -> float:
    idx = np.arange(len(psi), dtype=np.int64)
    s_grp = 0.0
    for g in groups:
        w = np.zeros_like(psi)
        for j in g:
            xm, zm, ny = masks(terms[j])
            w += coeffs[j] * apply(psi, idx, xm, zm, ny)
        ex = float(np.vdot(psi, w).real)
        var = max(float(np.vdot(w, w).real) - ex * ex, 0.0)
        s_grp += var ** 0.5
    return float((np.abs(coeffs) @ sigma) ** 2 / s_grp ** 2)


def fci_energy(prov: dict, basis: str) -> float:
    from pyscf import fci, gto, scf
    mol = gto.M(atom=[(a, (x, y, z)) for a, x, y, z in prov["geometry_angstrom"]], basis=basis,
                charge=int(prov["charge"]), spin=int(prov["multiplicity"]) - 1, verbose=0)
    mf = scf.RHF(mol).run()
    return float(fci.FCI(mf).kernel()[0])


def main() -> int:
    rows_main = [json.loads(line) for line in open(ROOT / "results/raw/main.jsonl") if line.strip()]
    out = []
    checks = {}
    for name in INSTANCES:
        t0 = time.perf_counter()
        inst = load(name)
        pset = PauliSet.from_terms(inst.terms, inst.coeffs)
        terms = list(inst.terms)
        coeffs = np.asarray(pset.coeffs, dtype=float)
        assert len(terms) == pset.m, "instance terms must already be clean"
        n_el = int(inst.provenance["n_electrons"])
        e0, psi, dsec = ground_state(terms, coeffs, inst.n_qubits, n_el)
        e_qubit = e0 + float(inst.identity_coeff)
        e_fci = fci_energy(inst.provenance, inst.basis)
        idx = np.arange(len(psi), dtype=np.int64)
        expval = np.array([np.vdot(psi, apply(psi, idx, *masks(t))).real for t in terms])
        sigma = np.sqrt(np.clip(1.0 - expval ** 2, 0.0, None))
        checks[name] = {"sector_dim": dsec, "E_qubit": e_qubit, "E_fci_pyscf": e_fci, "abs_diff": abs(e_qubit - e_fci)}
        for rel in ("QWC", "FC"):
            sel = {r["method"]: r for r in rows_main if r.get("status") == "ok" and r["instance"] == name
                   and r["relation"] == rel and r.get("seed", 0) == 0 and r.get("ablation", "none") == "none"}
            for meth in METHODS:
                if meth == "dtp_before_polish":
                    b = sel["dtp"]
                    res = run_dtp(pset, rel, seed=0, ig_budget=int(b["x_ig_steps"]),
                                  iteration_budget=int(b["x_tabu_iterations"]), do_polish=False)
                    colours = list(res.colours)
                    if res.k != int(b["x_k_after_reduce"]) or abs(res.rhat - float(b["x_rhat_before_polish"])) > 1e-9:
                        raise RuntimeError(f"replay of {name} {rel} does not reproduce the recorded Phase 3 output")
                else:
                    colours = sel[meth]["colours"]
                groups = groups_from_colours(np.asarray(colours, dtype=np.int64))
                rh = float(rhat(groups, coeffs))
                if meth != "dtp_before_polish" and abs(rh - float(sel[meth]["rhat"])) > 1e-9:
                    raise RuntimeError(f"R-hat of {name} {rel} {meth} differs from the recorded value")
                out.append({"instance": name, "relation": rel, "method": meth, "k": len(groups), "R_hat": rh,
                            "R_psi": r_psi(groups, terms, coeffs, psi, sigma)})
        print(f"{name}: sector {dsec}, E(qubit) - E(FCI) = {e_qubit - e_fci:+.2e} Ha, {time.perf_counter() - t0:.1f} s",
              flush=True)
    df = pd.DataFrame(out)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/tables/state_dependent.csv", index=False)
    with open(ROOT / "results/tables/state_dependent_checks.json", "w") as fh:
        json.dump(checks, fh, indent=1)
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
