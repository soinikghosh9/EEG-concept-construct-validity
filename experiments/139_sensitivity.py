"""Sensitivity of the headline results to three choices the text currently states without support.

Each of these is fixed somewhere in the code and justified nowhere in the manuscript, and each is
the kind of thing a reviewer asks about precisely because a favourable value could have been
chosen after seeing the outcome:

  * **the L2 strength, C = 0.5.** For a sign-constrained model this is not an ordinary
    regularisation knob. Coefficients on constrained coordinates are bounded below at zero, so C
    determines how many concepts are driven to exactly zero -- that is, it determines which
    quantities appear in the clinical report. It sets the interpretation, not just the fit.
  * **the 20 Hz analysis ceiling.** Justified a priori from the electromyographic literature, but
    the validity verdicts must not sit on a knife edge at exactly that value.
  * **the subject aggregation statistic.** The median was chosen for robustness to contaminated
    windows; whether the conclusions depend on that choice has never been tested.

A result that moves smoothly across a parameter is reported with its sensitivity curve. A result
that flips at one particular value is not a result.

    python experiments/139_sensitivity.py --mode channel
Out: outputs/results/audit/sensitivity_<mode>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, safe_auc  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEED, FOLDS = 42, 5
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}
CLINICAL_DIRECTION = {
    "pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
    "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
    "asymmetry_alpha": 0, "asymmetry_freq": 0,
    "theta_alpha_ratio": +1, "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
    "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
    "artefact_proxy": 0, "log_n_windows": 0,
}
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")
C_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
AGG = ("median", "mean", "trimmed20")


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def aggregate(W, how):
    if how == "median":
        return np.nanmedian(W, axis=0)
    if how == "mean":
        return np.nanmean(W, axis=0)
    lo, hi = np.nanpercentile(W, [20, 80], axis=0)
    M = np.where((W >= lo) & (W <= hi), W, np.nan)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(M, axis=0)
    return np.where(np.isfinite(out), out, np.nanmedian(W, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    ap.add_argument("--tag", default=None,
                    help="feature table to read, if not the one named by --mode. Lets an "
                         "alternative corpus be analysed without disturbing the table "
                         "other results were computed on.")

    args = ap.parse_args()
    tag = args.tag or args.mode
    OUT.mkdir(parents=True, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import StratifiedKFold

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)

    d = FEAT / tag

    # every array in this directory is joined by row index; a stale one
    # would silently pair features with the wrong windows
    check_table(d)

    C_win = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    names = [CONCEPT_NAMES[i] for i in cols]
    dirs = [CLINICAL_DIRECTION[n] for n in names]
    tr_all = sub_coh == "CAUEEG"
    report = {"mode": args.mode}

    def build(agg_how):
        C = np.stack([aggregate_subject(C_win[i]) if agg_how == "median"
                      else np.concatenate([[aggregate(C_win[i], agg_how)[0],
                                            aggregate(C_win[i], agg_how)[1]],
                                           [np.nanstd(C_win[i][:, 0])],
                                           aggregate(C_win[i], agg_how)[2:15],
                                           [np.log1p(len(i))]])
                      for i in idxs])
        C, _ = impute_from_training(C, tr_all)
        return C[:, cols]

    # ---------------------------------------------------------------- 1. C
    print(f"{'='*80}\n1. L2 strength C -- it decides how many concepts survive, so it decides "
          f"the report")
    C = build("median")
    csens = {}
    for site, pos, lab in (("ds004504_raw", DEM, "OpenNeuro CN-vs-dementia"),
                           ("GENEEG", MCI, "GENEEG CN-vs-MCI")):
        tr = tr_all & np.isin(y3, [CN, pos])
        te = (sub_coh == site) & np.isin(y3, [CN, pos])
        if te.sum() < 16:
            continue
        y_tr, y_te = (y3[tr] == pos).astype(int), (y3[te] == pos).astype(int)
        print(f"\n  {lab}")
        print(f"  {'C':>7s} {'AUC':>7s} {'nonzero concepts':>18s}  {'zeroed'}")
        rows = {}
        for c in C_GRID:
            m = SignConstrainedLogistic(dirs, C=c).fit(C[tr], y_tr)
            a = safe_auc(y_te, m.predict_proba(C[te])[:, 1])
            nz = int((np.abs(m.coef_standardised_) > 1e-8).sum())
            zeroed = [names[i] for i in range(len(names))
                      if abs(m.coef_standardised_[i]) <= 1e-8]
            rows[str(c)] = {"auc": a, "n_nonzero": nz, "zeroed": zeroed}
            print(f"  {c:7.2f} {fmt(a):>7s} {nz:18d}  "
                  f"{', '.join(zeroed[:4])}{'...' if len(zeroed) > 4 else ''}")
        csens[lab] = rows
    report["C_sensitivity"] = csens

    # ---------------------------------------------------------------- 2. EMG ceiling
    print(f"\n{'='*80}\n2. EMG analysis ceiling -- do the validity verdicts sit on a knife edge?")
    # This check has since been executed by 143_emg_sweep.py, which runs the extraction pass per
    # cutoff that cannot be done by refitting. Point at its output rather than restating a
    # placeholder, so a reader of this file is not told a completed check is outstanding.
    sweep = OUT / f"emg_sweep_{args.mode}.json"
    if sweep.exists():
        s = json.loads(sweep.read_text())
        report["emg_ceiling_sensitivity"] = {
            "status": "executed", "source": sweep.name, "grid_hz": s.get("grid_hz"),
            "verdict_stable_across_grid": s.get("verdict_stable_across_grid"),
            "cutoff_dependent": s.get("verdict_changes_across_grid")}
    else:
        report["emg_ceiling_sensitivity"] = {
            "status": "specified; run 143_emg_sweep.py after one extraction pass per cutoff",
            "grid_hz": [15, 20, 25, 30]}

    # ---------------------------------------------------------------- 3. aggregation
    print(f"\n{'='*80}\n3. Subject aggregation statistic")
    agg_rows = {}
    for how in AGG:
        Cx = build(how)
        line = {}
        for site, pos, lab in (("ds004504_raw", DEM, "OpenNeuro"), ("GENEEG", MCI, "GENEEG")):
            tr = tr_all & np.isin(y3, [CN, pos])
            te = (sub_coh == site) & np.isin(y3, [CN, pos])
            if te.sum() < 16:
                continue
            y_tr, y_te = (y3[tr] == pos).astype(int), (y3[te] == pos).astype(int)
            m = SignConstrainedLogistic(dirs, C=0.5).fit(Cx[tr], y_tr)
            line[lab] = safe_auc(y_te, m.predict_proba(Cx[te])[:, 1])
        agg_rows[how] = line
        print(f"  {how:12s} " + "  ".join(f"{k} {fmt(v)}" for k, v in line.items()))
    report["aggregation_sensitivity"] = agg_rows

    dest = write_result(OUT / f"sensitivity_{tag}.json", report,
                        script="139_sensitivity.py")
    print(f"\nwrote {dest}")
    print("\nThe C column to read is 'zeroed', not 'AUC': if the set of concepts appearing in the "
          "clinical\nreport changes materially across the grid, then the report is a function of "
          "a hyperparameter\nand that has to be stated in the paper.")


if __name__ == "__main__":
    main()
