"""What interpretability costs, decomposed on the harmonised corpus.

`119` answers this, but through `_audit_cache.load_corpus()` and the pre-harmonisation feature
files, so its evaluation site holds 88 subjects rather than the analysed 87 and its numbers
describe a corpus no table in the paper is computed on. Re-running it with provenance (done)
does not fix that; the data layer has to change, which is what this script does.

Seven arms, each differing from the previous one in exactly one respect, so the difference between
consecutive arms prices that one respect:

    A0  window-level random forest on spectral features, subject score by mean posterior
    A1  the same forest trained and scored per subject          -> training-unit protocol
    A2  a subject-level forest on the fourteen concepts         -> the concept bottleneck
    A3  subject-level logistic on spectral features             (dimensionality, reported aside)
    A4  subject-level logistic on the concepts                  -> linearity
    A5  the two-stage hierarchy on the concepts                 -> hierarchical structure
    A6  the same hierarchy with the clinical sign constraints   -> the constraints themselves

Only A5 minus A6 is the price of guaranteed clinical direction. A0 minus A1 is a protocol
difference and is not a cost of being interpretable; reporting the total as though it were is the
error this decomposition exists to prevent.

    python experiments/169_cost_decomposition_harmonised.py
Out: outputs/results/audit/cost_decomposition_harmonised.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEED, FOLDS, N_BOOT = 42, 5, 5000

DIRS = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
        "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
        "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
        "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
        "aperiodic_exponent": 0, "spectral_edge": -1}
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.data.feature_table import check_table
    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)
    from src.utils.provenance import write_result
    from _protocol_v2 import safe_auc

    d = FEAT / args.tag
    check_table(d)
    C_win = np.load(d / "window_concepts.npy")
    S_win = np.load(d / "spectral.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    coh = np.array([cohort[i[0]] for i in idxs])
    lab = np.array([str(labels[i[0]]).upper() for i in idxs])

    CN_LAB = {"CN", "C", "CONTROL", "NORMAL"}
    DEM_LAB = {"AD", "A", "FTD", "F"}
    is_cn, is_dem = np.isin(lab, list(CN_LAB)), np.isin(lab, list(DEM_LAB))
    y_imp = (~is_cn).astype(int)                       # level 1: control vs any impairment
    y_dem = is_dem.astype(int)                         # level 2: dementia among impaired

    tr = coh == "CAUEEG"
    site = coh == "ds004504_raw"
    keep = site & (is_cn | is_dem)                     # the dementia contrast at the blind site
    y_eval = is_dem[keep].astype(int)
    print("train %d subjects | evaluate %d at the dementia site (%d dementia)"
          % (tr.sum(), keep.sum(), y_eval.sum()))

    C_sub = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    C_sub, _ = impute_from_training(C_sub, tr)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    names = [CONCEPT_NAMES[i] for i in cols]
    Z = C_sub[:, cols]
    dirs = [DIRS[n] for n in names]
    Ssub = np.stack([S_win[i].mean(axis=0) for i in idxs])

    def rf():
        return RandomForestClassifier(n_estimators=400, n_jobs=6, random_state=SEED,
                                      class_weight="balanced")

    def lr():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))

    def subject_arm(model_fn, X):
        """Fit on the training cohort under subject-grouped folds, score the blind site."""
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
        p = []
        for a, _ in skf.split(X[tr], y_imp[tr]):
            m = model_fn().fit(X[tr][a], y_imp[tr][a])
            p.append(m.predict_proba(X[keep])[:, 1])
        return np.mean(p, axis=0)

    # ---- A0: window-level forest on spectral, subject score by mean posterior ----------------
    w_tr = np.isin(groups, subs[tr])
    y_win = np.repeat(y_imp, [len(i) for i in idxs])[np.argsort(np.concatenate(idxs))]
    sgkf = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    w_idx = np.flatnonzero(w_tr)
    site_w = np.flatnonzero(np.isin(groups, subs[keep]))
    pw = []
    for a, _ in sgkf.split(np.zeros(len(w_idx)), y_win[w_idx], groups[w_idx]):
        m = rf().fit(S_win[w_idx[a]], y_win[w_idx[a]])
        pw.append(m.predict_proba(S_win[site_w])[:, 1])
    pw = np.mean(pw, axis=0)
    order = {s: k for k, s in enumerate(subs[keep])}
    A0 = np.zeros(int(keep.sum()))
    cnt = np.zeros(int(keep.sum()))
    for j, g in enumerate(groups[site_w]):
        A0[order[g]] += pw[j]; cnt[order[g]] += 1
    A0 /= np.maximum(cnt, 1)
    print("  A0 window-level forest fitted")

    arms = {"A0_window_RF_spectral": A0,
            "A1_subject_RF_spectral": subject_arm(rf, Ssub),
            "A2_subject_RF_concepts": subject_arm(rf, Z),
            "A3_subject_logistic_spectral": subject_arm(lr, Ssub),
            "A4_subject_logistic_concepts": subject_arm(lr, Z)}
    for k in list(arms)[1:]:
        print("  %s fitted" % k)

    # ---- A5 / A6: the two-stage hierarchy, unconstrained then sign-constrained --------------
    imp = y_imp.astype(bool)

    def hierarchy(constrained):
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)

        def make():
            return (SignConstrainedLogistic(dirs, C=0.5) if constrained
                    else SignConstrainedLogistic([0] * len(dirs), C=0.5))
        f1 = [make().fit(Z[tr][a], y_imp[tr][a]) for a, _ in skf.split(Z[tr], y_imp[tr])]
        m2 = tr & imp
        f2 = [make().fit(Z[m2][a], y_dem[m2][a]) for a, _ in skf.split(Z[m2], y_dem[m2])]
        p1 = np.mean([m.predict_proba(Z[keep])[:, 1] for m in f1], axis=0)
        p2 = np.mean([m.predict_proba(Z[keep])[:, 1] for m in f2], axis=0)
        return p1 * p2                                  # P(impaired) x P(dementia | impaired)

    arms["A5_hier_logistic_concepts"] = hierarchy(False)
    arms["A6_hier_signconstrained_concepts"] = hierarchy(True)
    print("  hierarchy arms fitted")

    auc = {k: safe_auc(y_eval, v) for k, v in arms.items()}
    print("\narm AUCs at the dementia site")
    for k, v in auc.items():
        print("  %-38s %.4f" % (k, v))

    dec = {
        "training_unit_A0_minus_A1": auc["A0_window_RF_spectral"] - auc["A1_subject_RF_spectral"],
        "bottleneck_A1_minus_A2": auc["A1_subject_RF_spectral"] - auc["A2_subject_RF_concepts"],
        "linearity_A2_minus_A4": (auc["A2_subject_RF_concepts"]
                                  - auc["A4_subject_logistic_concepts"]),
        "hierarchy_A4_minus_A5": (auc["A4_subject_logistic_concepts"]
                                  - auc["A5_hier_logistic_concepts"]),
        "sign_constraints_A5_minus_A6": (auc["A5_hier_logistic_concepts"]
                                         - auc["A6_hier_signconstrained_concepts"]),
        "dimensionality_A3_minus_A4": (auc["A3_subject_logistic_spectral"]
                                       - auc["A4_subject_logistic_concepts"]),
        "total_A0_minus_A6": auc["A0_window_RF_spectral"] - auc["A6_hier_signconstrained_concepts"],
    }
    print("\ndecomposition")
    for k, v in dec.items():
        print("  %-38s %+.4f" % (k, v))
    chain = sum(dec[k] for k in ("training_unit_A0_minus_A1", "bottleneck_A1_minus_A2",
                                 "linearity_A2_minus_A4", "hierarchy_A4_minus_A5",
                                 "sign_constraints_A5_minus_A6"))
    print("  %-38s %+.4f  (must equal the total)" % ("chain sum", chain))
    assert abs(chain - dec["total_A0_minus_A6"]) < 1e-9, "decomposition does not telescope"

    # interval on the term that matters, by paired bootstrap on the same subjects
    rng = np.random.default_rng(SEED)
    a5, a6, n = arms["A5_hier_logistic_concepts"], arms["A6_hier_signconstrained_concepts"], len(y_eval)
    ds = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y_eval[i])) < 2:
            continue
        x, z = safe_auc(y_eval[i], a5[i]), safe_auc(y_eval[i], a6[i])
        if x is not None and z is not None:
            ds.append(x - z)
    ds = np.asarray(ds)
    p = float(min(2.0 * min((ds <= 0).mean(), (ds >= 0).mean()), 1.0))
    ci = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    print("\nsign constraints: %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f"
          % (dec["sign_constraints_A5_minus_A6"], ci[0], ci[1], max(p, 1.0 / len(ds))))

    report = {"tag": args.tag, "n_train": int(tr.sum()), "n_eval": int(keep.sum()),
              "n_eval_dementia": int(y_eval.sum()), "arm_auc": auc, "decomposition": dec,
              "sign_constraint_ci95": ci, "sign_constraint_p": max(p, 1.0 / len(ds)),
              "note": "Supersedes cost_decomposition.json, which is computed through the "
                      "pre-harmonisation corpus (88 subjects at the dementia site, not 87)."}
    dest = write_result(OUT / "cost_decomposition_harmonised.json", report,
                        script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
