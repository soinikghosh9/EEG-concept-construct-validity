"""Dedicated binary heads, and site-wise standardisation, against the within-site ceiling.

Two findings from `132_geneeg_diagnosis.py` motivate this:

  * the control-versus-MCI contrast IS present at GENEEG -- a model trained and tested within that
    site reaches 0.713, so the cross-site figure of 0.58 is a transfer loss of roughly 0.15, not an
    absence of signal;
  * a binary control-versus-MCI head reached 0.644 at GENEEG where the three-class hierarchical
    model reached 0.582, and per-site standardisation lifted the spectral model from 0.428 to
    0.571.

Both are one-line changes that were never tested together. Pooling mild cognitive impairment into
a three-class objective lets the gradient be dominated by the easier dementia contrast; asking the
model the question it will actually be asked is the obvious correction, and standardising each
site by its own moments corrects a pure location and scale shift using no labels from the target.

This script crosses them, for both clinical targets, both feature sets, three model families and
five seeds, and reports every result against the within-site ceiling rather than against chance --
because "0.64" means something quite different when the achievable maximum is 0.71 than when it is
1.0.

Site standardisation uses only the target cohort's FEATURE distribution, never its labels, so it
is available at deployment: a clinic can compute its own means and standard deviations from
unlabelled recordings before any patient is diagnosed.

    python experiments/133_binary_heads.py
Out: outputs/results/audit/binary_heads.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, auc_ci, safe_auc  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEEDS = (42, 43, 44, 45, 46)
FOLDS = 5
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}

#: (target label, blind cohort, three-class reference value from 128_benchmark_harmonised).
TARGETS = {
    "CN_vs_MCI": {"pos": MCI, "site": "GENEEG",
                  "three_class": {"14 concepts": 0.582, "123 spectral": 0.581}},
    "CN_vs_dementia": {"pos": DEM, "site": "ds004504_raw",
                       "three_class": {"14 concepts": 0.816, "123 spectral": 0.863}},
}

CLINICAL_DIRECTION = {
    "pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
    "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
    "asymmetry_alpha": 0, "asymmetry_freq": 0,
    "theta_alpha_ratio": +1, "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
    "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
    "artefact_proxy": 0, "log_n_windows": 0,
}
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def paired_bootstrap(y, a, b, n_boot=5000, seed=42):
    y = np.asarray(y)
    a0, b0 = safe_auc(y, a), safe_auc(y, b)
    if a0 is None or b0 is None:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    a, b, n = np.asarray(a), np.asarray(b), len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        da, db = safe_auc(y[i], a[i]), safe_auc(y[i], b[i])
        if da is not None and db is not None:
            out.append(da - db)
    d = np.asarray(out)
    return (a0 - b0, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)

    d = FEAT / args.mode

    # every array in this directory is joined by row index; a stale one
    # would silently pair features with the wrong windows
    check_table(d)

    C_win = np.load(d / "window_concepts.npy")
    S_win = np.load(d / "spectral.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    C_all = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    C_all, _ = impute_from_training(C_all, sub_coh == "CAUEEG")
    S = np.stack([np.median(S_win[i], axis=0) for i in idxs])
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    dirs = [CLINICAL_DIRECTION[CONCEPT_NAMES[i]] for i in cols]
    C = C_all[:, cols]
    cau = sub_coh == "CAUEEG"
    print(f"mode={args.mode} | {len(subs)} subjects")

    report = {"mode": args.mode, "seeds": list(SEEDS), "targets": {}}

    for tname, spec in TARGETS.items():
        pos, site = spec["pos"], spec["site"]
        tr_m = cau & np.isin(y3, [CN, pos])
        te_m = (sub_coh == site) & np.isin(y3, [CN, pos])
        y_tr = (y3[tr_m] == pos).astype(int)
        y_te = (y3[te_m] == pos).astype(int)
        print(f"\n{'='*80}\n{tname} -> {site}   train n={int(tr_m.sum())}  test n={int(te_m.sum())}")

        # within-site ceiling, for context
        ceil = {}
        for fname, F in (("123 spectral", S), ("14 concepts", C)):
            best = 0.5
            for proto in (RandomForestClassifier(n_estimators=400, random_state=42,
                                                 class_weight="balanced", n_jobs=6),
                          make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000,
                                                           class_weight="balanced"))):
                oof = np.zeros(len(y_te))
                for a, b in StratifiedKFold(FOLDS, shuffle=True,
                                            random_state=42).split(F[te_m], y_te):
                    m = clone(proto).fit(F[te_m][a], y_te[a])
                    oof[b] = m.predict_proba(F[te_m][b])[:, 1]
                best = max(best, safe_auc(y_te, oof) or 0.5)
            ceil[fname] = best
        print(f"  within-site ceiling: " +
              "  ".join(f"{k} {v:.3f}" for k, v in ceil.items()))

        rows = {}
        scores = {}
        for fname, F in (("123 spectral", S), ("14 concepts", C)):
            Ftr_raw, Fte_raw = F[tr_m], F[te_m]
            # Per-site standardisation: each cohort by its OWN moments. No target labels used.
            Ftr_z = (Ftr_raw - Ftr_raw.mean(0)) / (Ftr_raw.std(0) + 1e-9)
            Fte_z = (Fte_raw - Fte_raw.mean(0)) / (Fte_raw.std(0) + 1e-9)

            models = {"random forest": lambda s: RandomForestClassifier(
                          n_estimators=400, random_state=s, class_weight="balanced", n_jobs=6),
                      "logistic": lambda s: make_pipeline(
                          StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))}
            if fname == "14 concepts":
                models["sign-constrained"] = lambda s: SignConstrainedLogistic(dirs, C=0.5)

            for mname, mk in models.items():
                for stag, (A, B) in (("raw", (Ftr_raw, Fte_raw)),
                                     ("site-standardised", (Ftr_z, Fte_z))):
                    per = []
                    last = None
                    for s in SEEDS:
                        m = mk(s).fit(A, y_tr)
                        p = m.predict_proba(B)[:, 1]
                        per.append(safe_auc(y_te, p))
                        last = p
                    v = np.array([x for x in per if x is not None])
                    key = f"{fname} | {mname} | {stag}"
                    a, lo, hi = auc_ci(y_te, last)
                    rows[key] = {"auc_mean": float(v.mean()), "auc_sd": float(v.std(ddof=1)),
                                 "per_seed": v.tolist(), "ci95_last_seed": [lo, hi],
                                 "ceiling": ceil[fname],
                                 "fraction_of_ceiling": float(v.mean() / ceil[fname])}
                    scores[key] = last
                    print(f"  {key:48s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}  "
                          f"({100*v.mean()/ceil[fname]:.0f}% of ceiling)")

        # paired tests against the three-class reference for the same feature set
        best_key = max(rows, key=lambda k: rows[k]["auc_mean"])
        print(f"\n  best arm: {best_key}  ({rows[best_key]['auc_mean']:.3f})")
        base_key = next((k for k in rows if k.startswith("14 concepts")
                         and "sign-constrained" in k and "raw" in k), None)
        if base_key and base_key != best_key:
            dd, lo, hi, p = paired_bootstrap(y_te, scores[best_key], scores[base_key])
            print(f"  vs sign-constrained raw: {dd:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
            rows[best_key]["vs_signconstrained_raw"] = {"delta": dd, "ci95": [lo, hi], "p": p}

        report["targets"][tname] = {"n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
                                    "site": site, "ceiling": ceil,
                                    "three_class_reference": spec["three_class"], "arms": rows}

    dest = write_result(OUT / "binary_heads.json", report, script="133_binary_heads.py")
    print(f"\nwrote {dest}")
    print("\nRead every value against the within-site ceiling, not against chance. A cross-site "
          "AUC of\n0.64 where the achievable maximum is 0.71 is a different statement from the "
          "same number\nwhere the maximum is 1.0, and only the first is honest about what "
          "remains to be gained.")


if __name__ == "__main__":
    main()
