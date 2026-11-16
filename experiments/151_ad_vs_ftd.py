"""Alzheimer's against frontotemporal dementia: the subtype question, answered where it can be.

The paper has so far reported this as not evaluable, on the grounds that the training cohort holds
14 frontotemporal subjects. That is the right verdict for TRANSFER -- nothing useful can be trained
on 14 subjects and applied elsewhere -- but it is the wrong verdict for the question itself. The
dementia site contains 36 Alzheimer's and 22 frontotemporal subjects, which supports a within-site
cross-validated estimate. Small, but reportable with an interval, and honest about what it is.

Two arms, because they answer different questions and only one of them is a deployment claim:

  * **within-site**, 5-fold subject-grouped cross-validation at the dementia site. Estimates
    whether the contrast is present in the signal at all. Not a transfer result and not evidence
    the model would work anywhere else.
  * **transfer**, trained on the training cohort's 12 frontotemporal subjects and applied to the
    site unchanged. Reported so the reader can see what a real deployment of this would look like,
    which is the more informative number precisely because it is the worse one.

Everything runs on the de-confounded corpus: common 235 s span, exactly 24 windows per subject, so
neither recording length nor window count can separate the classes.

    python experiments/151_ad_vs_ftd.py --tag channel_clean
Out: outputs/results/audit/ad_vs_ftd_<tag>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import safe_auc                    # noqa: E402
from src.utils.provenance import write_result        # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")
AD_LABELS = {"AD", "A"}
FTD_LABELS = {"FTD", "F"}
SEED, FOLDS = 42, 5


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = safe_auc(y[i], s[i])
        if a is not None:
            out.append(a)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--site", default="ds004504_raw")
    ap.add_argument("--train-cohort", default="CAUEEG")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.models.v6 import CONCEPT_NAMES, aggregate_subject, impute_from_training

    d = FEAT / args.tag

    # every array in this directory is joined by row index; a stale one
    # would silently pair features with the wrong windows
    check_table(d)

    Cw = np.load(d / "window_concepts.npy")
    Sw = np.load(d / "spectral.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    g, lab, coh = meta["groups"], meta["labels"], meta["cohort"]
    if len(Sw) != len(g):
        raise SystemExit(f"spectral rows {len(Sw)} != windows {len(g)}; stale feature file")

    subs = np.unique(g)
    idxs = [np.flatnonzero(g == s) for s in subs]
    lab_s = np.array([lab[i[0]] for i in idxs])
    coh_s = np.array([coh[i[0]] for i in idxs])
    C = np.stack([aggregate_subject(Cw[i]) for i in idxs])
    S = np.nan_to_num(np.stack([np.nanmedian(Sw[i], axis=0) for i in idxs]))
    C, _ = impute_from_training(C, coh_s == args.train_cohort)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    Xc = C[:, cols]

    is_ad = np.isin(lab_s, list(AD_LABELS))
    is_ftd = np.isin(lab_s, list(FTD_LABELS))
    site = coh_s == args.site
    tr_coh = coh_s == args.train_cohort

    n_site_ad, n_site_ftd = int((site & is_ad).sum()), int((site & is_ftd).sum())
    n_tr_ad, n_tr_ftd = int((tr_coh & is_ad).sum()), int((tr_coh & is_ftd).sum())
    print(f"{args.site}: {n_site_ad} AD, {n_site_ftd} FTD")
    print(f"{args.train_cohort}: {n_tr_ad} AD, {n_tr_ftd} FTD")
    report = {"tag": args.tag, "site": args.site,
              "n_site_ad": n_site_ad, "n_site_ftd": n_site_ftd,
              "n_train_ad": n_tr_ad, "n_train_ftd": n_tr_ftd, "arms": {}}

    if min(n_site_ad, n_site_ftd) < 8:
        raise SystemExit("too few subjects at the site for any estimate")

    reps = {"concepts": Xc, "spectral": S}
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=500, random_state=SEED, min_samples_leaf=2),
    }

    # ---------------------------------------------------------------- within-site CV
    print(f"\n{'='*76}\nWithin-site cross-validation (NOT a transfer result)\n{'='*76}")
    sel = np.flatnonzero(site & (is_ad | is_ftd))
    y = is_ftd[sel].astype(int)
    grp = subs[sel]
    sgkf = StratifiedGroupKFold(min(FOLDS, int(min(np.bincount(y)))), shuffle=True,
                                random_state=SEED)
    for rep, X in reps.items():
        for mname, mk in models.items():
            oof = np.full(len(sel), np.nan)
            for a, b in sgkf.split(X[sel], y, grp):
                m = mk().fit(X[sel][a], y[a])
                oof[b] = m.predict_proba(X[sel][b])[:, 1]
            auc = safe_auc(y, oof)
            lo, hi = boot_ci(y, oof)
            report["arms"][f"within_site|{rep}|{mname}"] = {
                "auc": auc, "ci95": [lo, hi], "n": int(len(y)),
                "note": "within-site cross-validation; not evidence of transfer"}
            print(f"  {rep:9s} {mname:15s} AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]  n={len(y)}")

    # ---------------------------------------------------------------- transfer
    print(f"\n{'='*76}\nTransfer from {args.train_cohort} "
          f"({n_tr_ftd} FTD available to train on)\n{'='*76}")
    if min(n_tr_ad, n_tr_ftd) < 8:
        msg = (f"only {n_tr_ftd} frontotemporal subjects in {args.train_cohort}; a transfer "
               f"estimate would be uninterpretable and is not computed")
        print(f"  {msg}")
        report["arms"]["transfer"] = {"status": "not computed", "reason": msg}
    else:
        trm = np.flatnonzero(tr_coh & (is_ad | is_ftd))
        y_tr = is_ftd[trm].astype(int)
        for rep, X in reps.items():
            for mname, mk in models.items():
                m = mk().fit(X[trm], y_tr)
                sc = m.predict_proba(X[sel])[:, 1]
                auc = safe_auc(y, sc)
                lo, hi = boot_ci(y, sc)
                report["arms"][f"transfer|{rep}|{mname}"] = {
                    "auc": auc, "ci95": [lo, hi], "n_train": int(len(y_tr)),
                    "n_test": int(len(y))}
                print(f"  {rep:9s} {mname:15s} AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]")

    write_result(OUT / f"ad_vs_ftd_{args.tag}.json", report, script=Path(__file__).name)
    print(f"\nwrote {OUT / f'ad_vs_ftd_{args.tag}.json'}")


if __name__ == "__main__":
    main()
