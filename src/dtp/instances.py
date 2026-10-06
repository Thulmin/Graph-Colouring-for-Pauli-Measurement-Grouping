"""Instance generation: molecular qubit Hamiltonians as lists of Pauli terms.

Pipeline (pre-registration, section 4):
  1. PySCF restricted Hartree-Fock (RHF) for the molecule, basis and geometry;
  2. MO one- and two-electron integrals -> OpenFermion ``InteractionOperator``
     (spin orbitals interleaved alpha/beta, OpenFermion convention);
  3. Jordan-Wigner (``jordan_wigner``) or Bravyi-Kitaev (``bravyi_kitaev``) transform;
  4. coefficients are checked to be real (|imag| <= 1e-6; observed numerical noise is ~1e-8),
     the imaginary noise is discarded, terms with |Re a| < 1e-8 are dropped,
     and the identity term is removed (its coefficient is stored separately).

Each instance is cached as JSON in ``data/instances/<name>.json`` together with its provenance:
geometry and its source, charge, multiplicity, basis, encoding, threshold, qubit count n,
term counts, coefficient norms, SCF energy, software versions, generation time and the
SHA-256 hash of the canonical term list (one line ``<pauli string> <coefficient %.17g>`` per term,
terms sorted by string).

Pauli strings use index q = qubit q (OpenFermion convention), letters I/X/Y/Z.
Requires the optional ``chem`` dependencies (pyscf, openfermion, pyyaml).
Original code written for this project.
"""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ENCODINGS = ("JW", "BK")
THRESHOLD = 1e-8
IMAG_TOL = 1e-6
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "molecules.yaml"
DEFAULT_CACHE = ROOT / "data" / "instances"


@dataclass
class Instance:
    """A cleaned molecular Pauli-sum Hamiltonian with provenance."""

    name: str
    molecule: str
    basis: str
    encoding: str
    n_qubits: int
    terms: list[str]
    coeffs: list[float]
    identity_coeff: float
    provenance: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        """Number of non-identity terms."""
        return len(self.terms)

    def to_json(self) -> dict:
        """Plain-dict representation written to the JSON cache."""
        return asdict(self)


def instance_name(molecule: str, basis: str, encoding: str) -> str:
    """Canonical file stem, e.g. ``H2O_sto-3g_JW``."""
    return f"{molecule}_{basis.lower()}_{encoding.upper()}"


def term_list_sha256(terms: list[str], coeffs: list[float]) -> str:
    """SHA-256 of the canonical term list (sorted by string, coefficients as %.17g)."""
    h = hashlib.sha256()
    for s, c in sorted(zip(terms, coeffs)):
        h.update(f"{s} {float(c):.17g}\n".encode("utf-8"))
    return h.hexdigest()


def load_molecules(config: Path | str = DEFAULT_CONFIG) -> dict:
    """Read configs/molecules.yaml (geometries, charges, multiplicities and their sources)."""
    import yaml
    with open(config, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def qubit_operator_to_terms(qop, n_qubits: int, threshold: float = THRESHOLD,
                            imag_tol: float = IMAG_TOL) -> tuple[list[str], list[float], float, dict]:
    """Convert an OpenFermion ``QubitOperator`` into (strings, real coefficients, identity, stats)."""
    terms: list[str] = []
    coeffs: list[float] = []
    identity = 0.0
    max_imag = 0.0
    dropped = 0
    raw = 0
    for term, c in qop.terms.items():
        raw += 1
        c = complex(c)
        max_imag = max(max_imag, abs(c.imag))
        if term == ():
            identity += c.real
            continue
        if abs(c.real) < threshold:
            dropped += 1
            continue
        s = ["I"] * n_qubits
        for q, p in term:
            s[q] = p
        terms.append("".join(s))
        coeffs.append(float(c.real))
    if max_imag > imag_tol:
        raise ValueError(f"imaginary coefficient part {max_imag:.3e} exceeds {imag_tol:g}")
    order = sorted(range(len(terms)), key=terms.__getitem__)
    terms = [terms[i] for i in order]
    coeffs = [coeffs[i] for i in order]
    stats = {"m_raw_including_identity": raw, "dropped_below_threshold": dropped,
             "max_abs_imag": max_imag}
    return terms, coeffs, identity, stats


def generate(molecule: str, spec: dict, basis: str = "sto-3g", encoding: str = "JW",
             threshold: float = THRESHOLD) -> Instance:
    """Run the PySCF -> OpenFermion pipeline for one molecule (``spec`` from molecules.yaml)."""
    import openfermion as of
    import pyscf
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import ao2mo, gto, scf

    enc = encoding.upper()
    if enc not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    t0 = time.perf_counter()
    atom = [(a, (float(x), float(y), float(z))) for a, x, y, z in spec["geometry"]]
    mol = gto.M(atom=atom, basis=basis, charge=int(spec.get("charge", 0)),
                spin=int(spec.get("multiplicity", 1)) - 1, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    e_scf = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {molecule}/{basis}")
    C = mf.mo_coeff
    norb = C.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)
    h2 = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
    one, two = spinorb_from_spatial(h1, h2)
    iop = of.InteractionOperator(mol.energy_nuc(), one, 0.5 * two)
    n = 2 * norb
    if enc == "JW":
        qop = of.jordan_wigner(iop)
    else:
        qop = of.bravyi_kitaev(of.get_fermion_operator(iop), n_qubits=n)
    terms, coeffs, identity, stats = qubit_operator_to_terms(qop, n, threshold)
    a = np.asarray(coeffs)
    prov = {
        "geometry_angstrom": spec["geometry"],
        "geometry_source": spec.get("source", ""),
        "charge": int(spec.get("charge", 0)),
        "multiplicity": int(spec.get("multiplicity", 1)),
        "n_spatial_orbitals": norb,
        "n_electrons": int(mol.nelectron),
        "scf_energy_hartree": e_scf,
        "nuclear_repulsion_hartree": float(mol.energy_nuc()),
        "threshold": threshold,
        "coeff_l1": float(np.abs(a).sum()),
        "coeff_l2": float(np.sqrt((a ** 2).sum())),
        "sha256": term_list_sha256(terms, coeffs),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_seconds": time.perf_counter() - t0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "versions": {"pyscf": pyscf.__version__, "openfermion": of.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
        **stats,
    }
    return Instance(instance_name(molecule, basis, enc), molecule, basis.lower(), enc, n,
                    terms, coeffs, identity, prov)


def save(inst: Instance, cache: Path | str = DEFAULT_CACHE) -> Path:
    """Write an instance to ``<cache>/<name>.json`` and return the path."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{inst.name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(inst.to_json(), fh, indent=1)
    return path


def load(name: str, cache: Path | str = DEFAULT_CACHE, verify: bool = True) -> Instance:
    """Load a cached instance; with ``verify`` the stored SHA-256 is recomputed and checked."""
    path = Path(cache) / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    inst = Instance(**d)
    if verify:
        h = term_list_sha256(inst.terms, inst.coeffs)
        if h != inst.provenance.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {name}: file may have been modified")
    return inst


def get(molecule: str, basis: str = "sto-3g", encoding: str = "JW",
        cache: Path | str = DEFAULT_CACHE, config: Path | str = DEFAULT_CONFIG) -> Instance:
    """Load from the cache, generating and saving the instance first if it is missing."""
    name = instance_name(molecule, basis, encoding)
    if (Path(cache) / f"{name}.json").exists():
        return load(name, cache)
    spec = load_molecules(config)[molecule]
    inst = generate(molecule, spec, basis, encoding)
    save(inst, cache)
    return inst
