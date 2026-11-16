"""Reliability, abstention and threshold transport, recomputed on the harmonised corpus.

All three were measured on the previous pipeline, and all three are load-bearing:

  * the manuscript RETRACTS a proposed mechanism on the basis of the reliability result, so that
    result has to hold on the corpus the rest of the paper uses;
  * abstention is the largest positive effect the project has measured, larger than any difference
    between model families;
  * threshold transport is the deployment procedure the paper recommends.

Three things are done differently from the earlier version, each because the earlier version
overstated something.

**The operating point is set from the SOURCE cohort's prevalence, not the target's.** The previous
analysis passed the evaluation site's true prevalence, which is label-derived; the protocol file
warns about this in its own docstring while the manuscript describes the procedure as label-free.
Both are reported here, and the label-free number is the primary one.

**The abstention signal is pre-specified.** Previously each model was shown with its own best
refusal signal, chosen after the fact from three to five candidates on 88 subjects. Here distance
from the operating threshold is the primary signal for every model, and the alternatives are
reported as exploratory.

**Split-half reliability is reported with the caveat that it is an upper bound.** Interleaved
halves of one recording share session, montage, alertness and technician; they bound test-retest
stability from above and are not a substitute for it.

Watch the direction of travel: roughly half the training cohort's variance was ocular and has now
been removed, so if the concept model still changes one screening decision in five, that instability
can no longer be attributed to artefact.

    python experiments/136_reliability_harmonised.py --mode channel
Out: outputs/results/audit/reliability_harmonised_<mode>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, TARGETS, pairwise_score, safe_auc, target_subset  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEED, FOLDS, MIN_WINDOWS = 42, 5, 8
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


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def icc21(a, b):
    Y = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 10:
        return float("nan")
    g = Y.mean()
    ss_r = k * ((Y.mean(1) - g) ** 2).sum()
    ss_c = n * ((Y.mean(0) - g) ** 2).sum()
    ss_e = ((Y - g) ** 2).sum() - ss_r - ss_c
    msr, msc = ss_r / (n - 1), ss_c / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else 0.0


def paired_icc(a1, a2, b1, b2, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(a1)
    d0 = icc21(a1, a2) - icc21(b1, b2)
    d = np.array([icc21(a1[i], a2[i]) - icc21(b1[i], b2[i])
                  for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    p = float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))
    return {"delta": d0, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": max(p, 1.0 / n_boot)}


def balanced_accuracy(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(np.unique(y)) < 2:
        return None
    return float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()))


def coverage_curve(y, score, thr, risk, grid=None):
    grid = grid if grid is not None else np.arange(1.0, 0.55, -0.1)
    order = np.argsort(risk)
    out = []
    for cov in grid:
        k = max(int(round(cov * len(y))), 10)
        keep = order[:k]
        b = (balanced_accuracy(y[keep], (score[keep] >= thr).astype(int))
             if len(np.unique(y[keep])) > 1 else None)
        out.append((float(cov), b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    ap.add_argument("--tag", default=None,
                    help="feature table to read, if not the one named by --mode. Lets an "
                         "alternative corpus (e.g. the de-confounded one) be analysed "
                         "without disturbing the table other results were computed on.")
    args = ap.parse_args()
    tag = args.tag or args.mode
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)

    d = FEAT / tag

    # every array in this directory is joined by row index; a stale one
    # would silently pair features with the wrong windows
    check_table(d)

    C_win = np.load(d / "window_concepts.npy")
    S_win = np.load(d / "spectral.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    ok = np.array([len(i) >= MIN_WINDOWS for i in idxs])
    subs, idxs = subs[ok], [i for i, k in zip(idxs, ok) if k]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    tr = sub_coh == "CAUEEG"
    print(f"mode={args.mode} | {len(subs)} subjects with >= {MIN_WINDOWS} windows")

    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    dirs = [CLINICAL_DIRECTION[CONCEPT_NAMES[i]] for i in cols]

    def concepts(sel):
        C = np.stack([aggregate_subject(C_win[i]) for i in sel])
        C, _ = impute_from_training(C, np.ones(len(C), bool))
        return C[:, cols]

    C_full = concepts(idxs)
    C_a = concepts([i[0::2] for i in idxs])
    C_b = concepts([i[1::2] for i in idxs])

    y_l1 = (y3 != CN).astype(int)
    y_l2 = (y3 == DEM).astype(int)
    imp = y3 != CN
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    f1 = [SignConstrainedLogistic(dirs, C=0.5).fit(C_full[tr][a], y_l1[tr][a])
          for a, _ in skf.split(C_full[tr], y3[tr])]
    f2 = [SignConstrainedLogistic(dirs, C=0.5).fit(C_full[tr & imp][a], y_l2[tr & imp][a])
          for a, _ in skf.split(C_full[tr & imp], y_l2[tr & imp])]

    def v6_impaired(C):
        p1 = np.mean([m.predict_proba(C)[:, 1] for m in f1], axis=0)
        return p1

    w_tr = np.flatnonzero(np.isin(groups, subs[tr]))
    y_win = np.zeros(len(groups), dtype=np.int64)
    for i, s in enumerate(idxs):
        y_win[s] = y3[i]
    rf_fits = []
    for a, _ in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(w_tr)), y_win[w_tr], groups[w_tr]):
        m = RandomForestClassifier(n_estimators=400, n_jobs=6, random_state=SEED,
                                   class_weight="balanced")
        m.fit(S_win[w_tr[a]], y_win[w_tr[a]]); rf_fits.append(m)

    def rf_impaired(sel):
        return np.array([1 - np.mean([r.predict_proba(S_win[i]) for r in rf_fits],
                                     axis=0).mean(0)[CN] for i in sel])

    report = {"mode": args.mode, "tag": tag, "note": "source-prevalence threshold is primary; the "
                                         "target-prevalence variant is an upper bound that uses "
                                         "target labels", "groups": {}}
    src_prev = float((y3[tr] != CN).mean())
    print(f"source (CAUEEG) impairment prevalence = {src_prev:.3f}\n")

    # ---------------- reliability ----------------
    for label, m in [("ALL", np.ones(len(subs), bool)), ("CAUEEG", tr)] + \
                    [(c, sub_coh == c) for c in ("ds004504_raw", "GENEEG")]:
        if m.sum() < 20:
            continue
        va, vb = v6_impaired(C_a[m]), v6_impaired(C_b[m])
        ra = rf_impaired([i[0::2] for i, k in zip(idxs, m) if k])
        rb = rf_impaired([i[1::2] for i, k in zip(idxs, m) if k])
        row = {"n": int(m.sum())}
        print(f"=== reliability {label} (n={int(m.sum())}) ===")
        for nm, (a, b) in (("concept", (va, vb)), ("forest", (ra, rb))):
            thr = float(np.quantile(np.concatenate([a, b]), 1 - src_prev))
            row[nm] = {"icc21_half": icc21(a, b),
                       "decision_flip_rate": float(((a >= thr) != (b >= thr)).mean()),
                       "threshold_source_prevalence": thr}
            print(f"  {nm:8s} ICC {row[nm]['icc21_half']:.3f} | "
                  f"decision flips {100*row[nm]['decision_flip_rate']:.1f}%")
        row["paired_icc"] = paired_icc(va, vb, ra, rb)
        print(f"  paired delta ICC {row['paired_icc']['delta']:+.3f} "
              f"p={row['paired_icc']['p']:.4f}\n")
        report["groups"][label] = row

    # ---------------- abstention + transport ----------------
    for site, target in (("ds004504_raw", "CN_vs_dementia"), ("GENEEG", "CN_vs_MCI")):
        m = sub_coh == site
        if m.sum() < 20:
            continue
        mt, yt = target_subset(y3[m], target)
        if mt.sum() < 16 or len(np.unique(yt)) < 2:
            continue
        lbl = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{target}"
        print(f"=== abstention and transport: {lbl} (n={int(mt.sum())}) ===")
        sel = [i for i, k in zip(idxs, m) if k]
        p1 = v6_impaired(C_full[m])
        p2 = np.mean([mm.predict_proba(C_full[m])[:, 1] for mm in f2], axis=0)
        P6 = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
        Pr = np.stack([np.mean([r.predict_proba(S_win[i]) for r in rf_fits],
                               axis=0).mean(0) for i in sel])
        srow = {"n": int(mt.sum())}
        for nm, P in (("concept", P6), ("forest", Pr)):
            sc = pairwise_score(P[mt], target)
            tgt_prev = float(yt.mean())
            thr_src = float(np.quantile(sc, 1 - src_prev))
            thr_tgt = float(np.quantile(sc, 1 - tgt_prev))
            oracle = max(balanced_accuracy(yt, (sc >= t).astype(int)) or 0
                         for t in np.quantile(sc, np.linspace(0.05, 0.95, 37)))
            # pre-specified refusal signal: distance from the operating threshold
            cur = coverage_curve(yt, sc, thr_src, -np.abs(sc - thr_src))
            srow[nm] = {
                "auc": safe_auc(yt, sc),
                "bacc_source_prevalence": balanced_accuracy(yt, (sc >= thr_src).astype(int)),
                "bacc_target_prevalence_UPPER_BOUND":
                    balanced_accuracy(yt, (sc >= thr_tgt).astype(int)),
                "bacc_oracle": float(oracle),
                "coverage_curve_uncertainty": cur,
            }
            b0 = srow[nm]["bacc_source_prevalence"]
            b60 = next((b for c, b in cur if abs(c - 0.6) < 1e-6), None)
            print(f"  {nm:8s} AUC {fmt(srow[nm]['auc'])} | bal.acc source-prev "
                  f"{fmt(b0)}  target-prev {fmt(srow[nm]['bacc_target_prevalence_UPPER_BOUND'])}"
                  f"  oracle {fmt(oracle)}")
            print(f"           abstention (distance from threshold): "
                  f"100% {fmt(b0)} -> 60% {fmt(b60)}")
        report.setdefault("sites", {})[lbl] = srow
        print()

    dest = write_result(OUT / f"reliability_harmonised_{tag}.json", report,
                        script="136_reliability_harmonised.py")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
