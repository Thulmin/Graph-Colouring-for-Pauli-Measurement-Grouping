"""Regenerate every results table, figure and the requirement-coverage report from raw results.

Inputs (read-only): results/raw/*.jsonl (one JSON row per run), data/instances/*.json,
results/tests/{junit.xml,coverage.json} (written by `pytest --junitxml ... --cov-report=json:...`),
configs/experiments.yaml.
Outputs: results/tables/*.csv and *.md, figures/*.png, results/summary.json.

Analysis definitions fixed before the evaluation results were inspected (pre-registration plus
docs/DEVIATIONS.md C6-C8):
* "evaluation instances" = the 11 main molecules under QWC and FC (22 pairs); H2/R7 use pairs with
  m >= 100; H3/R10 ratio uses QWC.
* per instance, DTP and other seeded methods are summarised by the median over seeds;
* R8 compares the per-instance median DTP k with the CP-SAT optimum (seed counts reported too);
* R12 uses the total in-process wall time of each timed DTP run with m <= 5,000 (main, bk, exact);
* R13 uses the child process's ru_maxrss for DTP runs with m <= 15,000;
* R19 fits log(Phase 0 + Phase 1 time) against log(m) over the per-instance medians of the timed
  QWC DTP runs with m >= 500 in the main and scaling sets;
* Wilcoxon signed-rank tests are exact (all 2^n sign flips, zero differences dropped, average ranks
  for ties), two-sided, run per relation; Holm correction over the four tests.

Usage: python experiments/analyse.py
Original code written for this project.
"""
from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"
CFG = yaml.safe_load(open(ROOT / "configs" / "experiments.yaml"))
DEPLOYED = ["qiskit_lf", "pennylane_lf", "pennylane_dsatur", "pennylane_rlf", "pennylane_gis"]
LABEL = {"dtp": "DTP", "iterated_greedy": "Iterated Greedy", "sorted_insertion": "Sorted Insertion",
         "random_greedy": "Random greedy", "smallest_last": "Smallest-last", "qiskit_lf": "Qiskit LF",
         "pennylane_lf": "PennyLane lf", "pennylane_dsatur": "PennyLane dsatur",
         "pennylane_rlf": "PennyLane rlf", "pennylane_gis": "PennyLane gis",
         "gcol_tabucol": "GCol TabuCol", "gcol_partialcol": "GCol PartialCol", "cpsat": "CP-SAT"}
MOLECULE = {"H2": "H₂", "LiH": "LiH", "BeH2": "BeH₂", "H2O": "H₂O", "NH3": "NH₃", "CH4": "CH₄", "N2": "N₂",
            "CO": "CO", "HCl": "HCl", "NaH": "NaH", "H2S": "H₂S", "HeH+": "HeH⁺", "H3+": "H₃⁺",
            "H4chain": "H₄ chain", "HF": "HF", "BH3": "BH₃"}
# colours follow the method (fixed order of the validated categorical palette, dataviz skill)
COLOUR = {"dtp": "#2a78d6", "iterated_greedy": "#eb6834", "sorted_insertion": "#1baf7a",
          "deployed": "#eda100", "random_greedy": "#e87ba4", "smallest_last": "#008300",
          "gcol": "#4a3aa7", "other": "#e34948"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NOTES: list[str] = []


# ----------------------------------------------------------------------------- loading
def read_phase(phase: str) -> pd.DataFrame:
    p = RAW / f"{phase}.jsonl"
    if not p.exists():
        NOTES.append(f"phase '{phase}' has no results file")
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(p) if line.strip()]
    df = pd.DataFrame(rows)
    if "ablation" not in df:
        df["ablation"] = "none"
    df["ablation"] = df["ablation"].fillna("none")
    if "variant" not in df:
        df["variant"] = ""
    df["variant"] = df["variant"].fillna("")
    df["molecule"] = df["instance"].str.split("_").str[0]
    # Observation O4: the workspace's memory control group (limit recorded in logs/kernel_oom.log) is below the runner's
    # 6 GiB limit, so a worker that outgrows it is killed by the kernel before the runner's own check and is recorded
    # as "error" without a message. Such rows (no message, sampled peak >= 95% of the group limit) are memory failures.
    if "status" in df and "child_peak_rss_bytes" in df:
        lim = cgroup_limit_bytes()
        if lim:
            note = df["note"].fillna("") if "note" in df else pd.Series("", index=df.index)
            os_kill = (df.status == "error") & note.isin(["", "error"]) & (df.child_peak_rss_bytes.fillna(0) >= 0.95 * lim)
            df.loc[os_kill, "status"] = "memory_os"
    # Observation O5: GCol's call of NetworkX's recursive clique heuristic exceeds Python's recursion limit on graphs
    # with cliques of more than about 1,000 vertices; such runs are errors with a known cause
    if "status" in df and "note" in df:
        rec = (df.status == "error") & df["note"].fillna("").str.contains("RecursionError")
        df.loc[rec, "status"] = "error_recursion"
    return df


def cgroup_limit_bytes() -> int | None:
    """Memory limit of the workspace's control group, as recorded in logs/kernel_oom.log (None if not recorded)."""
    p = ROOT / "logs" / "kernel_oom.log"
    if not p.exists():
        return None
    m = re.search(r"memory cgroup limit \(bytes\): (\d+)", p.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.get("status", pd.Series(dtype=str)) == "ok"] if len(df) else df


def mol_label(inst: str) -> str:
    mol, basis, enc = inst.split("_")
    lab = MOLECULE.get(mol, mol)
    if basis != "sto-3g":
        lab += f" ({basis.upper()})"
    if enc != "JW":
        lab += f" [{enc}]"
    return lab


def instance_meta() -> pd.DataFrame:
    from dtp.instances import load
    rows = []
    for set_name, specs in CFG["instance_sets"].items():
        for d in specs:
            name = f"{d['molecule']}_{d['basis']}_{d['encoding']}"
            try:
                inst = load(name, verify=True)
            except FileNotFoundError:
                continue
            rows.append({"set": set_name, "instance": name, "label": mol_label(name), "molecule": d["molecule"],
                         "basis": d["basis"], "encoding": d["encoding"], "n_qubits": inst.n_qubits, "m": inst.m,
                         "electrons": inst.provenance.get("n_electrons"),
                         "sha256_12": inst.provenance["sha256"][:12],
                         "geometry_source": inst.provenance.get("geometry_source", "")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def wilcoxon_exact(x, y) -> dict:
    """Exact two-sided Wilcoxon signed-rank test by enumerating all sign assignments."""
    from scipy.stats import rankdata
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return {"n_nonzero": 0, "W_plus": float("nan"), "p_exact": 1.0}
    r2 = (2 * rankdata(np.abs(d))).astype(int)        # doubled ranks are integers
    total = int(r2.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in r2:                                      # distribution of the positive-rank sum
        counts[r:] = counts[r:] + counts[:-r].copy() if r > 0 else counts
    probs = counts / counts.sum()
    w = int(r2[d > 0].sum())
    dev = abs(w - total / 2)
    support = np.arange(total + 1)
    p = float(probs[np.abs(support - total / 2) >= dev - 1e-9].sum())
    return {"n_nonzero": n, "W_plus": w / 2, "p_exact": min(1.0, p)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else float("nan")


# ----------------------------------------------------------------------------- tables
def write_table(df: pd.DataFrame, name: str, floatfmt: str = ".3f") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:  # tabulate missing: plain CSV-like fallback
        md = df.to_string(index=False)
    (TAB / f"{name}.md").write_text(md + "\n", encoding="utf-8")


def per_instance(main: pd.DataFrame, rel: str, meta: pd.DataFrame) -> pd.DataFrame:
    """Group counts per instance and method for one relation (main set, timed runs only)."""
    d = ok(main)
    d = d[(d["relation"] == rel) & (d["ablation"] == "none")]
    out = []
    for inst in [i for i in meta[meta.set == "main"].instance]:
        di = d[d.instance == inst]
        if di.empty:
            continue
        row = {"instance": inst, "label": mol_label(inst), "m": int(di.m.iloc[0])}
        dtp = di[di.method == "dtp"]
        if len(dtp):
            row.update({"LB": int(dtp.x_lb.median()), "DSATUR seed": int(dtp.x_k_seed.median()),
                        "DTP median": float(dtp.k.median()), "DTP min": int(dtp.k.min()),
                        "DTP max": int(dtp.k.max()), "DTP IQR": iqr(dtp.k), "DTP seeds": int(len(dtp))})
        for meth in DEPLOYED + ["smallest_last", "sorted_insertion"]:
            dm = di[di.method == meth]
            row[LABEL[meth]] = int(dm.k.iloc[0]) if len(dm) else np.nan
        for meth in ("random_greedy", "iterated_greedy"):
            dm = di[di.method == meth]
            row[f"{LABEL[meth]} median"] = float(dm.k.median()) if len(dm) else np.nan
            if meth == "iterated_greedy" and len(dm):
                row["Iterated Greedy min"] = int(dm.k.min())
                row["Iterated Greedy max"] = int(dm.k.max())
        dep = [row.get(LABEL[m_]) for m_ in DEPLOYED if not pd.isna(row.get(LABEL[m_], np.nan))]
        row["Best deployed"] = int(min(dep)) if dep else np.nan
        row["Deployed complete"] = len(dep) == len(DEPLOYED)
        if len(dtp) and dep:
            row["DTP - best deployed"] = row["DTP median"] - row["Best deployed"]
            row["DTP vs best deployed (%)"] = 100.0 * (row["DTP median"] - row["Best deployed"]) / row["Best deployed"]
        if len(dtp) and not pd.isna(row.get("Iterated Greedy median", np.nan)):
            row["DTP - IG"] = row["DTP median"] - row["Iterated Greedy median"]
        out.append(row)
    return pd.DataFrame(out)


def rhat_table(main: pd.DataFrame, rel: str, meta: pd.DataFrame) -> pd.DataFrame:
    d = ok(main)
    d = d[(d["relation"] == rel) & (d["ablation"] == "none")]
    out = []
    for inst in [i for i in meta[meta.set == "main"].instance]:
        di = d[d.instance == inst]
        dtp = di[di.method == "dtp"]
        if dtp.empty:
            continue
        ratio = dtp.rhat / dtp.x_rhat_before_polish
        row = {"instance": inst, "label": mol_label(inst), "m": int(dtp.m.iloc[0]),
               "R̂ DSATUR seed": float(dtp.x_rhat_seed.median()),
               "R̂ DTP before polish (A4)": float(dtp.x_rhat_before_polish.median()),
               "R̂ DTP": float(dtp.rhat.median()), "polish ratio (median)": float(ratio.median()),
               "polish ratio (min)": float(ratio.min()),
               "k never increased by polishing": bool((dtp.k <= dtp.x_k_after_reduce).all())}
        for meth in ("sorted_insertion", "pennylane_dsatur", "qiskit_lf"):
            dm = di[di.method == meth]
            row[f"R̂ {LABEL[meth]}"] = float(dm.rhat.iloc[0]) if len(dm) else np.nan
        dm = di[di.method == "iterated_greedy"]
        row["R̂ Iterated Greedy (median)"] = float(dm.rhat.median()) if len(dm) else np.nan
        si = di[di.method == "sorted_insertion"]
        if len(si):
            row["k Sorted Insertion"] = int(si.k.iloc[0])
            row["k DTP (median)"] = float(dtp.k.median())
            row["R̂ DTP / R̂ SI"] = row["R̂ DTP"] / row["R̂ Sorted Insertion"]
        out.append(row)
    return pd.DataFrame(out)


def ablation_table(main: pd.DataFrame, abl: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    base = ok(main)
    base = base[(base.relation == "QWC") & (base.method == "dtp") & (base.ablation == "none")]
    ig = ok(main)
    ig = ig[(ig.relation == "QWC") & (ig.method == "iterated_greedy")]
    a = ok(abl)
    out = []
    for inst in [i for i in meta[meta.set == "main"].instance]:
        b = base[base.instance == inst]
        if b.empty:
            continue
        # C16: ablations are compared with same-phase DTP control runs when they exist (main-phase runs otherwise)
        ctrl = a[(a.instance == inst) & (a.method == "dtp") & (a.ablation == "none")] if len(a) else a
        ref = ctrl if len(ctrl) else b
        row = {"instance": inst, "label": mol_label(inst), "m": int(b.m.iloc[0]),
               "DTP reference": "same-phase control (C16)" if len(ctrl) else "main phase",
               "DTP control runs": int(len(ctrl)),
               "DTP k": float(ref.k.median()), "DTP k (main phase)": float(b.k.median()),
               "A4 k (no polish)": float(b.x_k_after_reduce.median()),
               "DTP R̂": float(b.rhat.median()), "A4 R̂ (no polish)": float(b.x_rhat_before_polish.median()),
               "IG-only k (IG baseline)": float(ig[ig.instance == inst].k.median()) if len(ig[ig.instance == inst]) else np.nan,
               "DTP time to best (s)": float(np.median([t[-1][0] for t in ref.x_trajectory])),
               "DTP lazy gamma bytes (max)": int(ref.x_gamma_bytes_peak.max()),
               "DTP IG steps (median)": float(ref.x_ig_steps.median()),
               "DTP IG steps, main phase (median)": float(b.x_ig_steps.median())}
        for tag, title in (("A1", "A1 k (random seed)"), ("A2", "A2 k (no clique stop)"), ("A3", "A3 k (full gamma)"),
                           ("A5", "A5 k (fixed tenure 10)"), ("A6", "A6 k (TabuCol-only Phase 3)")):
            at = a[(a.instance == inst) & (a.ablation == tag)] if len(a) else a
            if len(at):
                row[title] = float(at.k.median())
                if tag == "A3":
                    row["A3 full gamma bytes (max)"] = int(at.x_gamma_bytes_peak.max())
                    row["A3 TabuCol iterations (median)"] = float(at.x_tabu_iterations.median())
                    row["DTP TabuCol iterations (median)"] = float(ref.x_tabu_iterations.median())
                if tag == "A2":
                    row["A2 time to best (s)"] = float(np.median([t[-1][0] for t in at.x_trajectory]))
            elif tag == "A2" and len(a):
                row[title] = "same as DTP (stop never fired)"
            else:
                row[title] = np.nan
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10,
        "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.axisbelow": True,
    })
    return plt


def fig_k_vs_deployed(tables: dict, plt) -> list[str]:
    """Relative difference to the best deployed option, per instance (QWC and FC panels)."""
    made = []
    series = [("DTP median", "DTP", COLOUR["dtp"]), ("Iterated Greedy median", "Iterated Greedy", COLOUR["iterated_greedy"]),
              ("Sorted Insertion", "Sorted Insertion", COLOUR["sorted_insertion"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    for ax, rel in zip(axes, ("QWC", "FC")):
        t = tables.get(f"main_k_{rel}")
        if t is None or t.empty:
            continue
        t = t[t.m >= 100].reset_index(drop=True)
        y = np.arange(len(t))
        h = 0.26
        for j, (col, lab, c) in enumerate(series):
            vals = 100.0 * (t[col] - t["Best deployed"]) / t["Best deployed"]
            ax.barh(y + (j - 1) * h, vals, height=h - 0.04, color=c, label=lab)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, t.label)
        ax.invert_yaxis()
        ax.set_xlabel("groups relative to best deployed option (%)")
        ax.set_title(f"{rel}: lower is better", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_k_vs_best_deployed.png"
    fig.savefig(out)
    plt.close(fig)
    made.append(out.name)
    return made


def fig_pareto(main: pd.DataFrame, plt) -> list[str]:
    """Group count versus R-hat, normalised to DTP, for QWC (one point per instance and method)."""
    d = ok(main)
    d = d[(d.relation == "QWC") & (d.ablation == "none") & (d.m >= 100)]
    if d.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    groups = [("sorted_insertion", "Sorted Insertion", COLOUR["sorted_insertion"]),
              ("iterated_greedy", "Iterated Greedy", COLOUR["iterated_greedy"]),
              ("pennylane_dsatur", "PennyLane dsatur", COLOUR["deployed"])]
    for meth, lab, c in groups:
        xs, ys = [], []
        for inst, di in d.groupby("instance"):
            base = di[di.method == "dtp"]
            dm = di[di.method == meth]
            if base.empty or dm.empty:
                continue
            xs.append(dm.k.median() / base.k.median())
            ys.append(dm.rhat.median() / base.rhat.median())
        ax.scatter(xs, ys, s=28, color=c, label=lab, edgecolors="white", linewidths=1.5, zorder=3)
    ax.scatter([1], [1], s=60, color=COLOUR["dtp"], label="DTP (reference)", edgecolors="white", linewidths=1.5, zorder=4)
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set_xlabel("groups relative to DTP (lower is better)")
    ax.set_ylabel("R̂ relative to DTP (higher is better)")
    ax.legend(fontsize=8, loc="best")
    out = FIG / "fig_pareto_qwc.png"
    fig.savefig(out)
    plt.close(fig)
    return [out.name]


def fig_trajectories(main: pd.DataFrame, plt) -> list[str]:
    d = ok(main)
    d = d[(d.ablation == "none") & (d.seed == 0) & d.method.isin(["dtp", "iterated_greedy"])]
    picks = [i for i in ("H2O_sto-3g_JW", "NH3_sto-3g_JW", "N2_sto-3g_JW", "CH4_sto-3g_JW") if i in set(d.instance)]
    if not picks:
        return []
    fig, axes = plt.subplots(2, len(picks), figsize=(1.9 * len(picks) + 1, 4.2), squeeze=False)
    for row, rel in enumerate(("QWC", "FC")):
        for col, inst in enumerate(picks):
            ax = axes[row][col]
            for meth, c, lab in (("dtp", COLOUR["dtp"], "DTP"), ("iterated_greedy", COLOUR["iterated_greedy"], "Iterated Greedy")):
                r = d[(d.instance == inst) & (d.relation == rel) & (d.method == meth)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if meth == "dtp":
                    traj = r.x_trajectory
                    t0 = r.x_timings["build"] + r.x_timings["seed"] + r.x_timings["bound"]
                    pts = [(t0 + t, k) for t, k in traj]
                    end = t0 + r.x_timings["reduce"]
                else:
                    traj = r.x_trajectory
                    t0 = r.x_setup_seconds
                    pts = [(t0 + t, k) for t, k in traj]
                    end = t0 + r.x_core_seconds
                xs = [p[0] for p in pts] + [end]
                ys = [p[1] for p in pts] + [pts[-1][1]]
                ax.step(xs, ys, where="post", color=c, linewidth=1.6, label=lab)
            ax.set_title(f"{mol_label(inst)} {rel}", loc="left", fontsize=8.5)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("groups k")
            if row == 1:
                ax.set_xlabel("time (s)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "fig_trajectories_seed0.png"
    fig.savefig(out)
    plt.close(fig)
    return [out.name]


def fig_scaling(fit: dict, pts: pd.DataFrame, plt) -> list[str]:
    if pts.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.scatter(pts.m, pts.phase01, s=30, color=COLOUR["dtp"], edgecolors="white", linewidths=1.5, zorder=3,
               label="per-instance median (QWC)")
    if fit.get("slope") is not None:
        xs = np.logspace(np.log10(pts.m.min()), np.log10(pts.m.max()), 50)
        ax.plot(xs, np.exp(fit["intercept"]) * xs ** fit["slope"], color=INK2, linewidth=1.2,
                label=f"fit: time = c * m^{fit['slope']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("terms m")
    ax.set_ylabel("Phase 0 + 1 wall time (s)")
    ax.legend(fontsize=8)
    out = FIG / "fig_scaling_phase01.png"
    fig.savefig(out)
    plt.close(fig)
    return [out.name]


def fig_memory(dtp_runs: pd.DataFrame, plt) -> list[str]:
    if dtp_runs.empty:
        return []
    g = dtp_runs.groupby("instance").agg(m=("m", "first"), rss=("ru_maxrss_kb", "max"),
                                        graph=("x_graph_bytes", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.scatter(g.m, g.rss / 1024, s=30, color=COLOUR["dtp"], edgecolors="white", linewidths=1.5, zorder=3,
               label="peak resident memory of the run")
    ax.scatter(g.m, g.graph / 2 ** 20, s=30, color=COLOUR["iterated_greedy"], edgecolors="white", linewidths=1.5,
               zorder=3, label="stored conflict graph")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("terms m")
    ax.set_ylabel("MiB")
    ax.legend(fontsize=8)
    out = FIG / "fig_memory.png"
    fig.savefig(out)
    plt.close(fig)
    return [out.name]


# ----------------------------------------------------------------------------- tests and coverage
def test_results() -> dict:
    out = {"junit": None, "coverage": None}
    j = ROOT / "results" / "tests" / "junit.xml"
    if j.exists():
        root = ET.parse(j).getroot()
        cases = []
        for tc in root.iter("testcase"):
            status = "passed"
            if tc.find("failure") is not None or tc.find("error") is not None:
                status = "failed"
            elif tc.find("skipped") is not None:
                status = "skipped"
            cases.append({"file": tc.get("classname"), "name": tc.get("name"), "status": status,
                          "seconds": float(tc.get("time", 0))})
        out["junit"] = pd.DataFrame(cases)
    c = ROOT / "results" / "tests" / "coverage.json"
    if c.exists():
        cov = json.load(open(c))
        files = {k: v["summary"]["percent_covered"] for k, v in cov["files"].items()}
        out["coverage"] = {"total": cov["totals"]["percent_covered"], "files": files}
    return out


def tests_pass(junit: pd.DataFrame | None, patterns: list[str]) -> tuple[bool | None, str]:
    if junit is None:
        return None, "junit report missing"
    sel = junit[junit.name.apply(lambda n: any(re.search(p, n) for p in patterns))]
    if sel.empty:
        return None, f"no tests match {patterns}"
    passed = int((sel.status == "passed").sum())
    return passed == len(sel), f"{passed}/{len(sel)} tests passed ({', '.join(sorted(set(sel.name)))[:300]})"


# ----------------------------------------------------------------------------- main
def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    meta = instance_meta()
    write_table(meta.drop(columns=["geometry_source"]), "instances")
    phases = {p: read_phase(p) for p in ("main", "g3", "bk", "exact", "ablation", "scaling", "gcol")}
    main_df = phases["main"]
    summary: dict = {"notes": NOTES}
    tables: dict[str, pd.DataFrame] = {}

    # ---- run accounting
    acct = []
    for p, df in phases.items():
        if df.empty:
            continue
        # a run skipped by the C10 rule (an earlier seed of the same method and instance hit a limit) is recorded with
        # status "memory" and skipped = True; it is counted separately because it never started
        st_col = df.get("status", pd.Series("ok", index=df.index)).astype(str)
        if "skipped" in df:
            st_col = st_col.where(~df["skipped"].fillna(False).astype(bool), "skipped")
        for (meth, st), g in df.groupby([df["method"], st_col]):
            acct.append({"phase": p, "method": meth, "status": st, "runs": len(g)})
    write_table(pd.DataFrame(acct), "run_accounting")

    # ---- main tables
    for rel in ("QWC", "FC"):
        if main_df.empty:
            break
        t = per_instance(main_df, rel, meta)
        tables[f"main_k_{rel}"] = t
        write_table(t, f"main_k_{rel}")
        r = rhat_table(main_df, rel, meta)
        tables[f"main_rhat_{rel}"] = r
        write_table(r, f"main_rhat_{rel}")

    # ---- validity of every run, all phases (gate G1/G2 and baselines)
    val = []
    for p, df in phases.items():
        d = ok(df)
        if d.empty:
            continue
        for meth, g in d.groupby("method"):
            complete = bool((g.colours.apply(len) == g.m).all())
            val.append({"phase": p, "method": meth, "runs": len(g), "valid": int(g.valid.sum()),
                        "complete": complete})
    validity = pd.DataFrame(val)
    write_table(validity, "validity")

    # ---- runtime per method (main, median wall seconds of the core call or whole run)
    if not main_df.empty:
        d = ok(main_df)
        d = d[d.ablation == "none"]
        rt = d.groupby(["instance", "relation", "method"]).wall_seconds.median().unstack("method").reset_index()
        rt.insert(1, "label", rt.instance.map(mol_label))
        write_table(rt, "runtime_main")

    # ---- ablations
    if not phases["ablation"].empty or not main_df.empty:
        at = ablation_table(main_df, phases["ablation"], meta)
        tables["ablation"] = at
        write_table(at, "ablation")

    # ---- GCol: independent TabuCol / PartialCol implementation (pre-registered baseline; descriptive table)
    gc = phases["gcol"]
    if not gc.empty:
        md_all = ok(main_df)
        md_all = md_all[md_all.ablation == "none"] if len(md_all) else md_all
        rows = []
        for (inst, rel), g in gc.groupby(["instance", "relation"]):
            m_known = g.m.dropna() if "m" in g else pd.Series(dtype=float)
            m_val = int(m_known.iloc[0]) if len(m_known) else int(meta.loc[meta.instance == inst, "m"].iloc[0])
            row = {"instance": inst, "label": mol_label(inst), "relation": rel, "m": m_val}
            for meth in ("gcol_tabucol", "gcol_partialcol"):
                gm = g[g.method == meth]
                good = gm[gm.status == "ok"] if "status" in gm else gm
                lab = LABEL[meth]
                row[f"{lab} runs ok"] = f"{len(good)}/{len(gm)}"
                row[f"{lab} median k"] = float(good.k.median()) if len(good) else np.nan
                row[f"{lab} median s"] = float(good.x_core_seconds.median()) if len(good) else np.nan
                skipped = gm["skipped"].fillna(False).astype(bool) if "skipped" in gm else pd.Series(False, index=gm.index)
                bad = sorted(set(gm[(gm.status != "ok") & ~skipped].status)) if "status" in gm else []
                row[f"{lab} failures"] = ", ".join(bad)
                row[f"{lab} not started"] = int(skipped.sum())
                if "child_peak_rss_bytes" in gm and gm.child_peak_rss_bytes.notna().any():
                    row[f"{lab} peak MiB"] = float(gm.child_peak_rss_bytes.max() / 2 ** 20)
            md = md_all[(md_all.instance == inst) & (md_all.relation == rel)] if len(md_all) else md_all
            dt = md[md.method == "dtp"] if len(md) else md
            ig_ = md[md.method == "iterated_greedy"] if len(md) else md
            row["DSATUR seed k"] = int(dt.x_k_seed.median()) if len(dt) else np.nan
            row["DTP median k"] = float(dt.k.median()) if len(dt) else np.nan
            row["Iterated Greedy median k"] = float(ig_.k.median()) if len(ig_) else np.nan
            rows.append(row)
        tables["gcol"] = pd.DataFrame(rows)
        write_table(tables["gcol"], "gcol")

    # ---- exact set
    ex = ok(phases["exact"])
    if not ex.empty:
        rows = []
        for (inst, rel), g in ex.groupby(["instance", "relation"]):
            cp = g[g.method == "cpsat"]
            dt = g[g.method == "dtp"]
            row = {"instance": inst, "label": mol_label(inst), "relation": rel, "m": int(g.m.iloc[0])}
            if len(cp):
                c = cp.iloc[0]
                row.update({"CP-SAT status": c.x_status, "CP-SAT k": int(c.x_objective),
                            "CP-SAT bound": c.x_best_bound, "CP-SAT seconds": c.x_core_seconds})
            if len(dt):
                row.update({"LB": int(dt.x_lb.median()), "DTP median": float(dt.k.median()),
                            "DTP seeds at optimum": int((dt.k == row.get("CP-SAT k", -1)).sum()) if len(cp) else np.nan,
                            "DTP seeds": int(len(dt))})
            dep = g[g.method.isin(DEPLOYED)]
            if len(dep):
                row["Best deployed"] = int(dep.k.min())
            si = g[g.method == "sorted_insertion"]
            if len(si):
                row["Sorted Insertion"] = int(si.k.iloc[0])
            rows.append(row)
        tables["exact"] = pd.DataFrame(rows)
        write_table(tables["exact"], "exact")

    # ---- BK set
    bk = ok(phases["bk"])
    if not bk.empty:
        rows = []
        for inst, g in bk.groupby("instance"):
            dt = g[g.method == "dtp"]
            row = {"instance": inst, "label": mol_label(inst), "m": int(g.m.iloc[0])}
            if len(dt):
                row.update({"LB": int(dt.x_lb.median()), "DSATUR seed": int(dt.x_k_seed.median()),
                            "DTP median": float(dt.k.median()), "DTP min": int(dt.k.min()), "DTP max": int(dt.k.max()),
                            "R5 DTP <= seed (all runs)": bool((dt.k <= dt.x_k_seed).all()),
                            "valid (all runs)": bool(dt.valid.all())})
            for meth in DEPLOYED + ["sorted_insertion"]:
                dm = g[g.method == meth]
                row[LABEL[meth]] = int(dm.k.iloc[0]) if len(dm) else np.nan
            ig = g[g.method == "iterated_greedy"]
            row["Iterated Greedy median"] = float(ig.k.median()) if len(ig) else np.nan
            if len(dt) and not pd.isna(row["Qiskit LF"]):
                row["R6 DTP median <= Qiskit LF"] = row["DTP median"] <= row["Qiskit LF"]
            rows.append(row)
        tables["bk"] = pd.DataFrame(rows)
        write_table(tables["bk"], "bk")

    # ---- G3 replays
    g3 = phases["g3"]
    if not g3.empty and not main_df.empty:
        base = ok(main_df)
        base = base[(base.method == "dtp") & (base.seed == 0) & (base.ablation == "none")]
        rows = []
        for (inst, rel), g in g3.groupby(["instance", "relation"]):
            b = base[(base.instance == inst) & (base.relation == rel)]
            if b.empty:
                continue
            h0 = b.colours_sha256.iloc[0]
            reps = g[g.variant.str.startswith("replay")]
            rep_ok = reps[reps.status == "ok"]
            timed = g[g.variant == "timed_repeat"]
            rows.append({"instance": inst, "label": mol_label(inst), "relation": rel,
                         "replays identical": f"{int((rep_ok.colours_sha256 == h0).sum())}/{len(reps)}",
                         "G3 pass": bool(len(reps) == 3 and (rep_ok.colours_sha256 == h0).sum() == 3),
                         "tracemalloc peak (MiB)": float(rep_ok.x_peak_traced_bytes.max() / 2 ** 20) if len(rep_ok) else np.nan,
                         "timed repeat identical (info)": bool((timed.colours_sha256 == h0).all()) if len(timed) else np.nan,
                         "timed repeat k": int(timed.k.iloc[0]) if len(timed) and "k" in timed else np.nan,
                         "seed-0 k": int(b.k.iloc[0])})
        tables["g3"] = pd.DataFrame(rows)
        write_table(tables["g3"], "g3_determinism")

    # ---- scaling and complexity fit (R19)
    sc = ok(phases["scaling"])
    dtp_timed = []
    for p in ("main", "scaling"):
        d = ok(phases[p])
        if d.empty:
            continue
        d = d[(d.method == "dtp") & (d.ablation == "none") & (d.variant == "")]
        dtp_timed.append(d)
    pts = pd.DataFrame()
    fit: dict = {"slope": None}
    if dtp_timed:
        allt = pd.concat(dtp_timed)
        q = allt[(allt.relation == "QWC") & (allt.m >= 500)].copy()
        if len(q):
            q["phase01"] = q.x_timings.apply(lambda t: t["build"] + t["seed"])
            pts = q.groupby("instance").agg(m=("m", "first"), phase01=("phase01", "median")).reset_index()
            if len(pts) >= 2:
                b1, b0 = np.polyfit(np.log(pts.m), np.log(pts.phase01), 1)
                fit = {"slope": float(b1), "intercept": float(b0), "points": int(len(pts))}
        if len(sc):
            srows = []
            for inst, g in sc[sc.method == "dtp"].groupby("instance"):
                tm = pd.DataFrame(list(g.x_timings))
                srows.append({"instance": inst, "label": mol_label(inst), "m": int(g.m.iloc[0]), "n": int(g.n.iloc[0]),
                              "density": float(g.x_density.iloc[0]), "graph mode": g.x_graph_mode.iloc[0],
                              "graph MiB": g.x_graph_bytes.iloc[0] / 2 ** 20,
                              "build s": tm.build.median(), "DSATUR s": tm.seed.median(), "bound s": tm.bound.median(),
                              "Phase 3 s": tm.reduce.median(), "polish s": tm.polish.median(), "check s": tm.check.median(),
                              "total s": tm.total.median(), "peak RSS MiB": g.ru_maxrss_kb.max() / 1024,
                              "k seed": int(g.x_k_seed.median()), "k median": float(g.k.median()),
                              "LB": int(g.x_lb.median()), "valid": bool(g.valid.all()), "runs": len(g)})
                si = sc[(sc.method == "sorted_insertion") & (sc.instance == inst)]
                srows[-1].update({"SI k": float(si.k.median()) if len(si) else np.nan,
                                  "SI s": float(si.wall_seconds.median()) if len(si) else np.nan,
                                  "SI valid": bool(si.valid.all()) if len(si) else np.nan})
            tables["scaling"] = pd.DataFrame(srows)
            write_table(tables["scaling"], "scaling")
    summary["complexity_fit"] = fit
    write_table(pts, "complexity_fit_points")

    # ---- statistics: exact Wilcoxon + Holm
    stats_rows = []
    for rel in ("QWC", "FC"):
        t = tables.get(f"main_k_{rel}")
        r = tables.get(f"main_rhat_{rel}")
        if t is not None and len(t):
            tt = t.dropna(subset=["DTP median", "Best deployed"])
            res = wilcoxon_exact(tt["DTP median"], tt["Best deployed"])
            rel_diff = ((tt["DTP median"] - tt["Best deployed"]) / tt["Best deployed"]).median()
            stats_rows.append({"test": f"{rel}: DTP median k vs best deployed k", "n_instances": len(tt),
                               **res, "median relative difference": float(rel_diff)})
        if r is not None and len(r):
            rr = r.dropna(subset=["R̂ DTP", "R̂ PennyLane dsatur"])
            res = wilcoxon_exact(rr["R̂ DTP"], rr["R̂ PennyLane dsatur"])
            rel_diff = ((rr["R̂ DTP"] - rr["R̂ PennyLane dsatur"]) / rr["R̂ PennyLane dsatur"]).median()
            stats_rows.append({"test": f"{rel}: DTP R̂ vs PennyLane dsatur R̂", "n_instances": len(rr),
                               **res, "median relative difference": float(rel_diff)})
    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        sdf["p_holm"] = holm(sdf.p_exact.tolist())
        tables["statistics"] = sdf
        write_table(sdf, "statistics", floatfmt=".4g")

    # ---- hypotheses
    hyp = []
    dtp_main = ok(main_df)
    dtp_main = dtp_main[(dtp_main.method == "dtp") & (dtp_main.ablation == "none")] if len(dtp_main) else dtp_main
    if len(dtp_main):
        h1 = bool((dtp_main.k <= dtp_main.x_k_seed).all())
        hyp.append({"id": "H1", "statement": "DTP k <= its DSATUR seed on 100% of runs (by construction)",
                    "result": f"{int((dtp_main.k <= dtp_main.x_k_seed).sum())}/{len(dtp_main)} runs", "holds": h1})
        both = pd.concat([tables[f"main_k_{r}"].assign(relation=r) for r in ("QWC", "FC") if f"main_k_{r}" in tables])
        b100 = both[both.m >= 100].dropna(subset=["DTP median", "Best deployed"])
        le = int((b100["DTP median"] <= b100["Best deployed"]).sum())
        lt = int((b100["DTP median"] < b100["Best deployed"]).sum())
        h2 = le == len(b100) and lt >= math.ceil(len(b100) / 2)
        hyp.append({"id": "H2", "statement": "median DTP k <= best deployed on every pair with m >= 100, and strictly lower on at least half",
                    "result": f"<= on {le}/{len(b100)}; < on {lt}/{len(b100)}", "holds": h2})
        rq = tables.get("main_rhat_QWC")
        allr = dtp_main
        never_inc = bool((allr.k <= allr.x_k_after_reduce).all())
        rhat_nondec = bool((allr.rhat >= allr.x_rhat_before_polish - 1e-12).all())
        med_ratio = float(rq["polish ratio (median)"].median()) if rq is not None and len(rq) else float("nan")
        hyp.append({"id": "H3", "statement": "polishing raises R̂: median ratio over QWC instances >= 1.5; k never increased",
                    "result": f"median ratio {med_ratio:.3f}; k never increased: {never_inc}; R̂ never decreased: {rhat_nondec}",
                    "holds": bool(med_ratio >= 1.5 and never_inc)})
        both_si = both.dropna(subset=["DTP median", "Sorted Insertion"])
        lt_si = int((both_si["DTP median"] < both_si["Sorted Insertion"]).sum())
        hyp.append({"id": "H4", "statement": "DTP uses fewer groups than Sorted Insertion on each evaluation pair",
                    "result": f"fewer on {lt_si}/{len(both_si)}", "holds": lt_si == len(both_si)})
        summary["H2_pairs"] = {"le": le, "lt": lt, "n": len(b100)}
        summary["H4_pairs"] = {"lt": lt_si, "n": len(both_si)}

    # ---- requirements R1-R20 and gate G1-G3
    tr = test_results()
    junit = tr["junit"]
    req = []

    def add(rid, text, met, evidence):
        req.append({"id": rid, "requirement": text, "met": met, "evidence": evidence})

    m1, e1 = tests_pass(junit, [r"test_reject_"])
    add("R1", "Reject malformed input", m1, e1)
    m2, e2 = tests_pass(junit, [r"test_empty_input", r"test_identity_only", r"test_duplicates_merged",
                                r"test_all_compatible", r"test_all_conflicting", r"test_reject_unequal_lengths"])
    add("R2", "Edge cases handled", m2, e2)
    for rid, rel in (("R3", "QWC"), ("R4", "FC")):
        d = dtp_main[dtp_main.relation == rel] if len(dtp_main) else dtp_main
        n_inst = d.instance.nunique() if len(d) else 0
        add(rid, f"Supports {rel}", bool(len(d) and d.valid.all() and n_inst == 11),
            f"{int(d.valid.sum()) if len(d) else 0}/{len(d)} DTP runs valid on {n_inst} {rel} instances")
    if len(dtp_main):
        add("R5", "Never worse than its seed", bool((dtp_main.k <= dtp_main.x_k_seed).all()),
            f"{int((dtp_main.k <= dtp_main.x_k_seed).sum())}/{len(dtp_main)} runs")
        both = pd.concat([tables[f"main_k_{r}"].assign(relation=r) for r in ("QWC", "FC")])
        okq = both.dropna(subset=["Qiskit LF"])
        r6 = int((okq["DTP median"] <= okq["Qiskit LF"]).sum())
        add("R6", "Never worse than the deployed default (Qiskit LF)", r6 == len(okq) and len(okq) == 22,
            f"{r6}/{len(okq)} instance-relation pairs")
        b100 = both[both.m >= 100].dropna(subset=["Best deployed"])
        lt = int((b100["DTP median"] < b100["Best deployed"]).sum())
        add("R7", "Strictly fewer groups than the best deployed option on >= 50% of pairs with m >= 100",
            lt >= math.ceil(0.5 * len(b100)), f"{lt}/{len(b100)} pairs")
    ex_t = tables.get("exact")
    if ex_t is not None and len(ex_t):
        solved = ex_t[ex_t["CP-SAT status"] == "OPTIMAL"]
        hit = int((solved["DTP median"] == solved["CP-SAT k"]).sum())
        add("R8", "Optimal where provable", hit == len(solved) and len(solved) > 0,
            f"median DTP k equals the CP-SAT optimum on {hit}/{len(solved)} solved instance-relation pairs "
            f"({len(ex_t) - len(solved)} not solved to optimality)")
    alld = []
    for p in ("main", "bk", "exact", "scaling"):
        d = ok(phases[p])
        if len(d):
            alld.append(d[(d.method == "dtp") & (d.ablation == "none") & (d.variant == "")])
    alld = pd.concat(alld) if alld else pd.DataFrame()
    if len(alld):
        has_lb = bool(alld.x_lb.notna().all())
        add("R9", "Certificate reported (LB and gap)", has_lb, f"LB recorded for {int(alld.x_lb.notna().sum())}/{len(alld)} runs")
        nondec = bool((dtp_main.rhat >= dtp_main.x_rhat_before_polish - 1e-12).all())
        rq = tables.get("main_rhat_QWC")
        med_ratio = float(rq["polish ratio (median)"].median()) if rq is not None and len(rq) else float("nan")
        add("R10", "Shot-cost awareness", nondec and med_ratio >= 1.5,
            f"R̂ never decreased by polishing: {nondec}; median QWC ratio {med_ratio:.3f}")
    if len(dtp_main):
        both = pd.concat([tables[f"main_k_{r}"].assign(relation=r) for r in ("QWC", "FC")])
        bs = both.dropna(subset=["Sorted Insertion"])
        lt_si = int((bs["DTP median"] < bs["Sorted Insertion"]).sum())
        add("R11", "Fewer groups than Sorted Insertion on >= 90% of pairs", lt_si >= math.ceil(0.9 * len(bs)),
            f"{lt_si}/{len(bs)} pairs")
    if len(alld):
        small = alld[(alld.m <= 5000) & alld.set.isin(["main", "bk", "exact"])]
        within = small.apply(lambda r: r.x_timings["total"] <= 1.1 * r.T_m, axis=1) if len(small) else pd.Series(dtype=bool)
        worst = small.apply(lambda r: r.x_timings["total"] / r.T_m, axis=1).max() if len(small) else float("nan")
        add("R12", "Runtime within T(m) + 10% (m <= 5,000)", bool(len(small) and within.all()),
            f"{int(within.sum())}/{len(small)} runs; worst total/T(m) = {worst:.3f}")
        mem = alld[alld.m <= 15000]
        worst_mem = mem.ru_maxrss_kb.max() / 1024 if len(mem) else float("nan")
        add("R13", "Peak resident memory <= 2 GB (m <= 15,000)", bool(len(mem) and (mem.ru_maxrss_kb * 1024 <= 2e9).all()),
            f"worst peak RSS {worst_mem:.0f} MiB over {len(mem)} runs")
        sc_d = alld[alld.set == "scaling"]
        if len(sc_d):
            biggest = sc_d[sc_d.m == sc_d.m.max()]
            ok14 = bool(biggest.valid.all() and (biggest.wall_seconds <= 1800).all())
            add("R14", "Valid grouping for the largest real instance within 30 minutes", ok14,
                f"{mol_label(biggest.instance.iloc[0])} (m = {int(biggest.m.iloc[0]):,}): {len(biggest)} runs, "
                f"longest {biggest.wall_seconds.max():.0f} s, " + ("all valid" if bool(biggest.valid.all()) else "not all valid"))
    # R15 is judged after this script finishes (all expected outputs present) - see below
    prov_cols = ["seed", "versions", "hardware", "instance_sha256"]
    okrows = pd.concat([ok(df) for df in phases.values() if len(df)]) if any(len(df) for df in phases.values()) else pd.DataFrame()
    if len(okrows):
        complete = bool(all(c in okrows for c in prov_cols) and okrows[prov_cols].notna().all().all())
        add("R16", "Provenance stored with every result row", complete,
            f"{len(okrows):,} result rows checked for {', '.join(prov_cols)}")
    cov = tr["coverage"]
    if cov:
        core = {k: v for k, v in cov["files"].items() if "/experiments/" not in k}
        add("R17", "Line coverage of the core package >= 80%", cov["total"] >= 80,
            f"total {cov['total']:.1f}% (core modules {min(core.values()):.0f}-{max(core.values()):.0f}%)")
    else:
        add("R17", "Line coverage of the core package >= 80%", None, "coverage.json missing")
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_docstrings import missing_docstrings
    missing, total = missing_docstrings()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sections = all(s in readme for s in ("## Install", "## Test", "## Reproduce", "## Use"))
    add("R18", "Documentation (README sections; docstrings)", sections and not missing,
        f"README sections present: {sections}; public objects without docstring: {len(missing)}/{total}")
    if fit.get("slope") is not None:
        add("R19", "Phase 0+1 log-log slope in [1.5, 2.5] (m >= 500)", 1.5 <= fit["slope"] <= 2.5,
            f"slope {fit['slope']:.3f} over {fit['points']} QWC instances")
    bk_t = tables.get("bk")
    if bk_t is not None and len(bk_t):
        r20 = bool(bk_t["R5 DTP <= seed (all runs)"].all() and bk_t["valid (all runs)"].all()
                   and bk_t["R6 DTP median <= Qiskit LF"].all())
        add("R20", "Encoding robustness (R5, R6, validity on BK)", r20, f"{len(bk_t)} BK instances")

    gates = []
    if len(alld):
        g_all = []
        for p in ("main", "bk", "exact", "scaling", "ablation", "g3"):
            d = ok(phases[p])
            if len(d):
                g_all.append(d[d.method == "dtp"])
        g_all = pd.concat(g_all)
        gates.append({"id": "G1", "gate": "Validity of every DTP output (independent checker)",
                      "pass": bool(g_all.valid.all()), "evidence": f"{int(g_all.valid.sum())}/{len(g_all)} DTP runs valid"})
        comp = bool((g_all.colours.apply(len) == g_all.m).all())
        gates.append({"id": "G2", "gate": "Completeness (every term in exactly one group)", "pass": comp and bool(g_all.valid.all()),
                      "evidence": "checker verifies exactly-once assignment; colour vector length equals m in every run"})
    g3t = tables.get("g3")
    if g3t is not None and len(g3t):
        gates.append({"id": "G3", "gate": "Determinism (3 deterministic replays per instance and relation)",
                      "pass": bool(g3t["G3 pass"].all()) and len(g3t) == 22,
                      "evidence": f"{int(g3t['G3 pass'].sum())}/{len(g3t)} instance-relation pairs with 3/3 identical replays"})
    gate_df = pd.DataFrame(gates)
    write_table(gate_df, "gates")

    # ---- figures (R15: generated from the raw files only)
    plt = setup_matplotlib()
    figs = []
    figs += fig_k_vs_deployed(tables, plt)
    figs += fig_pareto(main_df, plt)
    figs += fig_trajectories(main_df, plt)
    figs += fig_scaling(fit, pts, plt)
    if len(alld):
        figs += fig_memory(alld[alld.set.isin(["main", "scaling"]) & (alld.relation == "QWC")], plt)
    summary["figures"] = figs

    # ---- sensitivity analysis (not pre-registered; docs/DEVIATIONS.md O1)
    sens = []
    for rel in ("QWC", "FC"):
        t = tables.get(f"main_k_{rel}")
        if t is None or t.empty:
            continue
        sub = t[~t.instance.isin(["LiH_sto-3g_JW", "NH3_sto-3g_JW"])]
        b = sub[sub.m >= 100].dropna(subset=["Best deployed"])
        res = wilcoxon_exact(b["DTP median"], b["Best deployed"])
        sens.append({"relation": rel, "pairs (m >= 100, without LiH and NH3)": len(b),
                     "DTP < best deployed": int((b["DTP median"] < b["Best deployed"]).sum()),
                     "DTP <= best deployed": int((b["DTP median"] <= b["Best deployed"]).sum()),
                     "DTP < Iterated Greedy": int((sub["DTP median"] < sub["Iterated Greedy median"]).sum()),
                     "DTP > Iterated Greedy": int((sub["DTP median"] > sub["Iterated Greedy median"]).sum()),
                     "Wilcoxon p (exact, k vs best deployed)": res["p_exact"]})
    write_table(pd.DataFrame(sens), "sensitivity_without_LiH_NH3", floatfmt=".4g")

    # ---- R15 and the coverage percentage
    expected = ["main_k_QWC", "main_k_FC", "main_rhat_QWC", "main_rhat_FC", "ablation", "exact", "bk", "g3_determinism",
                "scaling", "statistics", "validity"]
    have = [e for e in expected if (TAB / f"{e}.csv").exists()]
    add("R15", "One command regenerates every table and figure from raw files", len(have) == len(expected) and len(figs) >= 5,
        f"{len(have)}/{len(expected)} tables and {len(figs)} figures written by experiments/analyse.py")
    req_df = pd.DataFrame(req)
    req_df["order"] = req_df.id.str[1:].astype(int)
    req_df = req_df.sort_values("order").drop(columns="order")
    write_table(req_df, "requirements")
    met = int((req_df.met == True).sum())  # noqa: E712
    summary["requirements"] = {"met": met, "assessed": int(req_df.met.notna().sum()), "denominator": 20,
                               "coverage_percent": 100.0 * met / 20, "target": "> 95% (20/20)"}
    hyp.append({"id": "H5", "statement": "requirement coverage > 95% (20/20)",
                "result": f"{met}/20 = {100.0 * met / 20:.1f}%", "holds": met == 20})
    write_table(pd.DataFrame(hyp), "hypotheses")
    summary["gates"] = gate_df.to_dict(orient="records")
    summary["hypotheses"] = hyp
    if junit is not None:
        summary["tests"] = {"total": len(junit), "passed": int((junit.status == "passed").sum()),
                            "failed": int((junit.status == "failed").sum()), "skipped": int((junit.status == "skipped").sum())}
        write_table(junit, "tests")
    if cov:
        summary["coverage_total"] = cov["total"]
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("requirements", "complexity_fit") if k in summary}, indent=1))
    print("figures:", figs)
    print("notes:", NOTES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
