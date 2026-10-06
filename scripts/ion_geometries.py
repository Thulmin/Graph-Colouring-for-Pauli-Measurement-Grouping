"""Compute equilibrium geometries for the two exact-set ions that have no experimental geometry
in NIST CCCBDB (HeH+ and H3+), and write them into configs/molecules.yaml.

Both ions have two electrons, so full configuration interaction (FCI) is exact within the
basis set. The bond length (HeH+) or the side of the equilateral triangle (H3+) is found by a
bounded scalar minimisation of the FCI/cc-pVTZ energy (Brent's method, tolerance 1e-6 Angstrom).

Usage:  python scripts/ion_geometries.py
Original code written for this project.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from pyscf import fci, gto, scf
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "molecules.yaml"
BASIS = "cc-pvtz"


def heh_geometry(r: float) -> list:
    return [["He", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, r]]


def h3_geometry(r: float) -> list:
    return [["H", 0.0, 0.0, 0.0], ["H", r, 0.0, 0.0], ["H", r / 2.0, r * math.sqrt(3.0) / 2.0, 0.0]]


def fci_energy(geometry: list, charge: int) -> float:
    atom = [(a, (x, y, z)) for a, x, y, z in geometry]
    mol = gto.M(atom=atom, basis=BASIS, charge=charge, spin=0, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    e, _ = fci.FCI(mf).kernel()
    return float(e)


def optimise(builder, bounds) -> tuple[float, float]:
    res = minimize_scalar(lambda r: fci_energy(builder(r), 1), bounds=bounds, method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x), float(res.fun)


def main() -> int:
    r_heh, e_heh = optimise(heh_geometry, (0.6, 1.0))
    r_h3, e_h3 = optimise(h3_geometry, (0.7, 1.1))
    print(f"HeH+  r_e = {r_heh:.6f} A   E_FCI/{BASIS} = {e_heh:.8f} Ha")
    print(f"H3+   R_e = {r_h3:.6f} A   E_FCI/{BASIS} = {e_h3:.8f} Ha")
    text = CONFIG.read_text()
    r1 = round(r_heh, 4)
    r2 = round(r_h3, 4)
    text = text.replace("R_HEH", f"{r1:.4f}")
    g = h3_geometry(r2)
    h3 = "[" + ", ".join(f"[H, {x:.4f}, {y:.4f}, {z:.4f}]" for _, x, y, z in g) + "]"
    text = text.replace("equilateral_R_H3", h3)
    text = re.sub(r"(computed equilibrium bond length \(FCI/cc-pVTZ scan)",
                  rf"computed equilibrium bond length {r1:.4f} A (FCI/cc-pVTZ scan", text)
    text = re.sub(r"(computed equilibrium side length of the equilateral triangle \(FCI/cc-pVTZ scan)",
                  rf"computed equilibrium side length {r2:.4f} A of the equilateral triangle (FCI/cc-pVTZ scan", text)
    CONFIG.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
