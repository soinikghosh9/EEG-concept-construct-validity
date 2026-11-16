"""The three deployment endpoints the paper promises and has never reported.

`main.tex` states design targets fixed before evaluation: probabilities calibrated to an expected
calibration error of 0.10 or better after installation; refusal of input unlike anything in the
training distribution; and a referral decision usable at a clinic absent from training. The
Results report none of the three. A reviewer can currently point at three unmet promises, and the
first paragraph's claim to be building a triage test is never tested as one.

  1. **Calibration.** Expected calibration error per site, before and after recalibration on a
     small labelled sample, with the reliability diagram's bins retained so the shape of the
     miscalibration is visible rather than summarised to a single number. Reported with the
     equal-mass binning that avoids the well-known instability of equal-width bins at n<100.

  2. **Out-of-distribution refusal.** Mahalanobis distance in concept space against the training
     cohort, thresholded at its 95th percentile. This closes the loop on the covariate-shift
     result: if GENEEG is refused essentially in full, then reporting its discrimination is
     reporting what happens when a guard is overridden, and the MCI section becomes a statement
     about a tool declining a site rather than failing on it.

  3. **Decision curves.** Net benefit across the range of referral thresholds a clinic might
     adopt, against treat-all and treat-none. AUC cannot answer "should this clinic use it"; net
     benefit can, and it is the standard instrument for exactly this question. The plausible
     threshold range should ultimately come from the clinical co-authors (worksheet D of the
     review guide); until then the full range is reported.

    python experiments/137_deployment_endpoints.py --mode channel
Out: outputs/results/audit/deployment_endpoints_<mode>.json
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


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def ece(y, p, n_bins=10):
    """Expected calibration error with EQUAL-MASS bins.

    Equal-width bins are the common choice and are unstable at these sample sizes: with n=88 and a
    model whose scores concentrate, several bins hold one or zero subjects and the statistic is
    dominated by them. Equal-mass bins put the same number of subjects in each.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    if n < 20:
        return None, []
    order = np.argsort(p)
    bins = np.array_split(order, min(n_bins, max(n // 8, 2)))
    e, rows = 0.0, []
    for b in bins:
        if len(b) == 0:
            continue
        conf, acc = float(p[b].mean()), float(y[b].mean())
        e += len(b) / n * abs(conf - acc)
        rows.append({"n": int(len(b)), "mean_predicted": conf, "observed": acc})
    return float(e), rows


def net_benefit(y, p, thresholds):
    """Vickers-Elkin net benefit, with treat-all and treat-none references."""
    y = np.asarray(y, int); p = np.asarray(p, float); n = len(y)
    out = []
    prev = y.mean()
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = t / max(1 - t, 1e-9)
        out.append({"threshold": float(t),
                    "net_benefit_model": float(tp / n - (fp / n) * w),
                    "net_benefit_treat_all": float(prev - (1 - prev) * w),
                    "net_benefit_treat_none": 0.0})
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

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
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
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    C_all = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    tr = sub_coh == "CAUEEG"
    C_all, _ = impute_from_training(C_all, tr)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    dirs = [CLINICAL_DIRECTION[CONCEPT_NAMES[i]] for i in cols]
    C = C_all[:, cols]
    print(f"mode={args.mode} | {len(subs)} subjects")

    report = {"mode": args.mode}

    # ------------------------------------------------------ 2. OOD guard (no training needed)
    print(f"\n{'='*78}\nOUT-OF-DISTRIBUTION GUARD (Mahalanobis in concept space, "
          f"CAUEEG 95th percentile)")
    mu, sd = C[tr].mean(0), C[tr].std(0) + 1e-9
    Z = (C - mu) / sd
    S = np.cov(Z[tr], rowvar=False)
    S += np.eye(len(S)) * 1e-3 * np.trace(S) / len(S)
    Si = np.linalg.inv(S)
    ctr = Z[tr].mean(0)

    def maha(A):
        D = A - ctr
        return np.sqrt(np.einsum("ij,jk,ik->i", D, Si, D))

    d_tr = maha(Z[tr])
    thr = float(np.percentile(d_tr, 95))
    ood = {"threshold_p95": thr, "cohorts": {}}
    print(f"  {'cohort':16s} {'n':>5s} {'median D':>9s} {'refused':>9s}")
    for c in sorted(set(sub_coh.tolist())):
        m = sub_coh == c
        if m.sum() < 10:
            continue
        dd = maha(Z[m])
        ood["cohorts"][c] = {"n": int(m.sum()), "median": float(np.median(dd)),
                             "refusal_rate": float((dd > thr).mean())}
        print(f"  {c:16s} {int(m.sum()):5d} {np.median(dd):9.2f} "
              f"{100*(dd > thr).mean():8.1f}%")
    report["ood_guard"] = ood

    # ------------------------------------------------------ models for 1 and 3
    y_l1 = (y3 != CN).astype(int)
    y_l2 = (y3 == DEM).astype(int)
    imp = y3 != CN
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    f1 = [SignConstrainedLogistic(dirs, C=0.5).fit(C[tr][a], y_l1[tr][a])
          for a, _ in skf.split(C[tr], y3[tr])]
    f2 = [SignConstrainedLogistic(dirs, C=0.5).fit(C[tr & imp][a], y_l2[tr & imp][a])
          for a, _ in skf.split(C[tr & imp], y_l2[tr & imp])]
    p1 = np.mean([m.predict_proba(C)[:, 1] for m in f1], axis=0)
    p2 = np.mean([m.predict_proba(C)[:, 1] for m in f2], axis=0)
    P6 = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])

    y_win = np.zeros(len(groups), dtype=np.int64)
    for i, s in enumerate(idxs):
        y_win[s] = y3[i]
    w_tr = np.flatnonzero(np.isin(groups, subs[tr]))
    rf_fits = []
    for a, _ in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(w_tr)), y_win[w_tr], groups[w_tr]):
        m = RandomForestClassifier(n_estimators=400, n_jobs=6, random_state=SEED,
                                   class_weight="balanced")
        m.fit(S_win[w_tr[a]], y_win[w_tr[a]]); rf_fits.append(m)
    Pr = np.stack([np.mean([r.predict_proba(S_win[i]) for r in rf_fits], axis=0).mean(0)
                   for i in idxs])

    rng = np.random.default_rng(SEED)
    for site, target in (("ds004504_raw", "CN_vs_dementia"), ("GENEEG", "CN_vs_MCI")):
        m = sub_coh == site
        mt, yt = target_subset(y3[m], target)
        if mt.sum() < 16 or len(np.unique(yt)) < 2:
            continue
        lbl = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{target}"
        print(f"\n{'='*78}\n{lbl}  (n={int(mt.sum())})")
        srow = {"n": int(mt.sum())}
        for nm, P in (("concept", P6), ("forest", Pr)):
            sc = pairwise_score(P[m][mt], target)
            e0, bins0 = ece(yt, sc)
            # recalibration on a small labelled sample, as the deployment procedure describes
            n_cal = min(20, len(yt) // 3)
            idx = rng.permutation(len(yt))
            cal, hold = idx[:n_cal], idx[n_cal:]
            iso = IsotonicRegression(out_of_bounds="clip").fit(sc[cal], yt[cal])
            e1, _ = ece(yt[hold], iso.predict(sc[hold]))
            nb = net_benefit(yt, sc, np.arange(0.05, 0.75, 0.05))
            better = sum(1 for r in nb
                         if r["net_benefit_model"] > max(r["net_benefit_treat_all"], 0.0))
            srow[nm] = {"auc": safe_auc(yt, sc), "ece": e0, "ece_bins": bins0,
                        "ece_after_recalibration": e1, "n_calibration": int(n_cal),
                        "net_benefit": nb,
                        "thresholds_where_model_wins": better,
                        "n_thresholds": len(nb)}
            print(f"  {nm:8s} AUC {fmt(srow[nm]['auc'])} | ECE {fmt(e0)} -> "
                  f"{fmt(e1)} after recalibrating on {n_cal} labelled subjects")
            print(f"           net benefit exceeds treat-all/treat-none at "
                  f"{better}/{len(nb)} thresholds")
        report.setdefault("sites", {})[lbl] = srow

    dest = write_result(OUT / f"deployment_endpoints_{tag}.json", report,
                        script="137_deployment_endpoints.py")
    print(f"\nwrote {dest}")
    print("\nIf a cohort's refusal rate is near 100 per cent, its discrimination figures describe "
          "what\nhappens when the guard is overridden, and should be reported that way rather "
          "than as a\nstraightforward failure of the model.")


if __name__ == "__main__":
    main()
