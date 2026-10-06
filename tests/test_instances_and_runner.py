"""Instance pipeline, provenance and experiment infrastructure (T25 and data-integrity checks)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from dtp.experiments.limits import run_limited
from dtp.instances import (Instance, instance_name, load, load_molecules, qubit_operator_to_terms, save,
                           term_list_sha256)

from conftest import ROOT, SMOKE_INSTANCE, requires_smoke


def test_instance_name_and_hash_are_canonical():
    assert instance_name("H2O", "STO-3G", "jw") == "H2O_sto-3g_JW"
    h1 = term_list_sha256(["ZI", "XX"], [0.5, 0.25])
    h2 = term_list_sha256(["XX", "ZI"], [0.25, 0.5])
    assert h1 == h2 and len(h1) == 64
    assert h1 != term_list_sha256(["ZI", "XX"], [0.5, 0.2500000001])


def test_save_load_and_tamper_detection(tmp_path):
    inst = Instance("T_sto-3g_JW", "T", "sto-3g", "JW", 2, ["XX", "ZI"], [0.25, 0.5], -1.0,
                    {"sha256": term_list_sha256(["XX", "ZI"], [0.25, 0.5])})
    p = save(inst, tmp_path)
    assert load("T_sto-3g_JW", tmp_path).terms == ["XX", "ZI"]
    d = json.loads(p.read_text())
    d["coeffs"][0] = 0.3
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="SHA-256"):
        load("T_sto-3g_JW", tmp_path)


def test_qubit_operator_conversion_rules():
    of = pytest.importorskip("openfermion")
    q = of.QubitOperator("", 1.5) + of.QubitOperator("X0 Z2", 0.3) + of.QubitOperator("Z1", 0.2 + 1e-9j)
    q.terms[((1, "Y"),)] = 1e-9      # set directly: OpenFermion's own addition already drops |c| < 1e-8
    terms, coeffs, ident, stats = qubit_operator_to_terms(q, 3)
    assert ident == 1.5 and terms == ["IZI", "XIZ"] and coeffs == [0.2, 0.3]
    assert stats["dropped_below_threshold"] == 1
    with pytest.raises(ValueError):
        qubit_operator_to_terms(of.QubitOperator("X0", 0.1 + 0.01j), 1)


def test_molecule_config_has_sources():
    mols = load_molecules()
    for name in ("H2", "LiH", "BeH2", "H2O", "NH3", "CH4", "N2", "CO", "HCl", "NaH", "H2S", "HF", "BH3",
                 "HeH+", "H3+", "H4chain"):
        assert name in mols and mols[name]["source"]
        assert all(len(atom) == 4 for atom in mols[name]["geometry"])


def test_h2_hamiltonian_ground_energy_matches_fci():
    pytest.importorskip("pyscf")
    of = pytest.importorskip("openfermion")
    from pyscf import fci, gto, scf
    from dtp.instances import generate
    spec = load_molecules()["H2"]
    inst = generate("H2", spec, "sto-3g", "JW")
    assert inst.m == 14 and inst.n_qubits == 4
    qop = of.QubitOperator((), inst.identity_coeff)
    for s, c in zip(inst.terms, inst.coeffs):
        qop += of.QubitOperator(" ".join(f"{p}{q}" for q, p in enumerate(s) if p != "I"), c)
    H = of.get_sparse_operator(qop, n_qubits=4).toarray()
    idx = [b for b in range(16) if bin(b).count("1") == 2]
    e_qubit = np.linalg.eigvalsh(H[np.ix_(idx, idx)])[0]
    mol = gto.M(atom=[(a, (x, y, z)) for a, x, y, z in spec["geometry"]], basis="sto-3g", verbose=0)
    e_fci = fci.FCI(scf.RHF(mol).run()).kernel()[0]
    assert e_qubit == pytest.approx(e_fci, abs=1e-9)


def test_run_limited_statuses():
    ok = run_limited([sys.executable, "-c", "print('hi')"], timeout_s=30)
    assert ok.status == "ok" and ok.stdout.strip() == "hi"
    err = run_limited([sys.executable, "-c", "raise SystemExit(3)"], timeout_s=30)
    assert err.status == "error" and err.returncode == 3
    slow = run_limited([sys.executable, "-c", "import time; time.sleep(10)"], timeout_s=0.5)
    assert slow.status == "timeout"
    big = run_limited([sys.executable, "-c", "import numpy as np, time; a = np.ones(400_000_000, np.uint8); time.sleep(5)"],
                      timeout_s=30, mem_bytes=100 * 2 ** 20)
    assert big.status == "memory"


# T25
@requires_smoke
@pytest.mark.parametrize("rel", ["QWC", "FC"])
def test_worker_end_to_end_smoke_instance(rel):
    job = {"job_id": f"test|HF|{rel}", "set": "test", "instance": SMOKE_INSTANCE, "relation": rel,
           "method": "dtp", "seed": 0, "budget": 2.0}
    res = run_limited([sys.executable, "-m", "dtp.experiments.worker", json.dumps(job)], timeout_s=300,
                      env={"PYTHONPATH": str(ROOT / "src")})
    assert res.status == "ok", res.stderr[-500:]
    row = json.loads(res.stdout.strip().splitlines()[-1])
    assert row["valid"] and row["k"] <= row["x_k_seed"] and row["k"] >= row["x_lb"]
    assert row["instance_sha256"] and row["versions"]["numpy"] and row["hardware"]["logical_cpus"]
    assert len(row["colours"]) == row["m"] == 630


@requires_smoke
def test_worker_runs_every_baseline_on_smoke_instance():
    from dtp.experiments.worker import run_job
    for method, extra in [("sorted_insertion", {}), ("smallest_last", {}), ("random_greedy", {}),
                          ("iterated_greedy", {"budget": 1.0}), ("qiskit_lf", {}), ("pennylane_dsatur", {}),
                          ("gcol_tabucol", {"it_limit": 200}), ("gcol_partialcol", {"it_limit": 200})]:
        row = run_job({"job_id": "t", "set": "test", "instance": SMOKE_INSTANCE, "relation": "FC",
                       "method": method, "seed": 0, **extra})
        assert row["valid"], method
    with pytest.raises(ValueError):
        run_job({"job_id": "t", "set": "test", "instance": SMOKE_INSTANCE, "relation": "FC", "method": "nope"})
