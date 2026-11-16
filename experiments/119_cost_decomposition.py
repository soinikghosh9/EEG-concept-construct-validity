"""R3: what is the cost of interpretability, once the comparison is controlled?

The manuscript's most-quoted number is a paired deficit of 0.109 AUC between the sign-constrained
concept model and a random forest at OpenNeuro, presented as the price of an auditable decision.
Four things differ between those two models, not one:

    axis                 concept model              random forest
    ------------------   ------------------------   -----------------------------
    training unit        1,120 SUBJECTS             ~62,000 WINDOWS, label copied
    features             14 named concepts          123 spectral features
    functional form      linear                     400 trees
    structure            hierarchical, 2 stages     flat 3-class

The first is the one that matters most and is least discussed. `src/models/v6/concepts.py:14`
argues -- in this repository -- that window-level training with the subject's label copied to every
window is pseudo-replication. The random forest does exactly that; the concept model does not. So
some unknown share of the 0.109 is a protocol difference rather than a cost of honesty, and a
reviewer will ask which.

This script fits the intermediate models that answer it, on identical folds and identical
subjects, and reports the ladder:

    A0  window-level RF, 123 spectral      the manuscript's incumbent (expect ~0.850)
    A1  subject-level RF, 123 spectral     A0 -> A1 isolates the TRAINING UNIT
    A2  subject-level RF, 14 concepts      A1 -> A2 isolates the BOTTLENECK
    A3  subject-level logistic, 123 spec   dimensionality at fixed model class
    A4  subject-level logistic, 14 conc    A2 -> A4 isolates LINEARITY
    A5  hierarchical logistic, 14 conc     A4 -> A5 isolates the HIERARCHY
    A6  hierarchical SIGN-CONSTRAINED      A5 -> A6 isolates the SIGN CONSTRAINTS
                                           (= the manuscript's model, expect 0.741)

Every arm is trained on CAUEEG only, scored at the blind sites with the restricted pairwise
posterior, and compared to A0 with the same subject-level paired bootstrap used elsewhere. The
honest "cost of the concept bottleneck" is A1 - A2; the cost of the interpretability machinery on
top of it is A2 - A6; the protocol difference is A0 - A1. Those three sum to the reported 0.109.

Nothing here is selected on the blind sites: every hyperparameter matches what the existing
experiments already use (RF 400 trees, logistic C=0.5, 5 folds, seed 42).

    python experiments/119_cost_decomposition.py
Out: outputs/results/audit/cost_decomposition.json
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

from _audit_cache import load_corpus, subject_table  # noqa: E402
from _protocol_v2 import (CN, DEM, TARGETS, TRAIN_COHORT, auc_ci,  # noqa: E402
                          evaluable_sites, pairwise_score, safe_auc, target_subset)

OUT = ROOT / "outputs" / "results" / "audit"
TA = ROOT / "outputs" / "results" / "track_a"
VAL = ROOT / "outputs" / "results" / "validity"
SPECTRAL_SLICE = slice(955, 1078)
SEED = 42
N_BOOT = 5000
#: Excluded from the classifier throughout: two declared confounds plus the concept the repair
#: could not fix. Identical to 110/111 so the A6 arm reproduces the published model exactly.
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")

CLINICAL_DIRECTION = {
    "pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
    "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
    "asymmetry_alpha": 0, "asymmetry_freq": 0,
    "theta_alpha_ratio": +1, "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
    "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
    "artefact_proxy": 0, "log_n_windows": 0,
}


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def paired_bootstrap(y, a, b, n_boot=N_BOOT, seed=SEED):
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
    p = float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))
    return a0 - b0, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.models.v6 import CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject

    c = load_corpus()
    groups, y3, coh, is_aug = c["groups"], c["y3"], c["cohort"], c["is_aug"]
    st = subject_table(c)
    subs, idxs, sub_y3, sub_coh = (st["subjects"], st["index"], st["y3"], st["cohort"])
    sites = evaluable_sites(set(sub_coh.tolist()))

    Xw = np.asarray(np.load(TA / "X_full.npy", mmap_mode="r")[:, SPECTRAL_SLICE],
                    dtype=np.float32)
    W = np.load(VAL / "window_concepts_emgfree.npy")
    C_full = np.stack([aggregate_subject(W[ix]) for ix in idxs])
    # Subject-level spectral features: the MEDIAN over the subject's real windows, matching how
    # the concept model aggregates. Using the mean here instead would introduce a second,
    # unrelated difference between the arms.
    Xsub = np.stack([np.median(Xw[ix], axis=0) for ix in idxs]).astype(np.float64)
    print(f"{len(subs)} subjects | spectral {Xsub.shape} | concepts {C_full.shape}")

    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    names = [CONCEPT_NAMES[i] for i in cols]
    dirs = [CLINICAL_DIRECTION[n] for n in names]
    Csub = C_full[:, cols]
    print(f"classifier concepts ({len(names)}): {', '.join(names)}\n")

    tr = sub_coh == TRAIN_COHORT
    imp = sub_y3 != CN
    y_l1 = (sub_y3 != CN).astype(int)
    y_l2 = (sub_y3 == DEM).astype(int)

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
    tr_folds = list(skf.split(Csub[tr], sub_y3[tr]))
    imp_folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
                     .split(Csub[tr & imp], y_l2[tr & imp]))

    # ------------------------------------------------------------------ arm builders
    def flat_subject(make, X):
        """Fit a flat 3-class model on CAUEEG subjects; return a scorer over all subjects."""
        fits = [make().fit(X[tr][a], sub_y3[tr][a]) for a, _ in tr_folds]
        classes = fits[0].classes_

        def score(Xq):
            P = np.mean([f.predict_proba(Xq) for f in fits], axis=0)
            full = np.zeros((len(Xq), 3))
            for k, cl in enumerate(classes):
                full[:, int(cl)] = P[:, k]
            return full
        return score

    def hierarchical_subject(make, X):
        """Two binary stages, recombined -- the structure the concept model uses."""
        f1 = [make().fit(X[tr][a], y_l1[tr][a]) for a, _ in tr_folds]
        f2 = [make().fit(X[tr & imp][a], y_l2[tr & imp][a]) for a, _ in imp_folds]

        def score(Xq):
            p1 = np.mean([m.predict_proba(Xq)[:, 1] for m in f1], axis=0)
            p2 = np.mean([m.predict_proba(Xq)[:, 1] for m in f2], axis=0)
            return np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
        return score

    def rf():
        return RandomForestClassifier(n_estimators=400, n_jobs=6, random_state=SEED,
                                      class_weight="balanced")

    def lr():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5))

    # A0: the incumbent -- window-level RF, subject score = mean of window posteriors.
    wtr = np.flatnonzero((coh == TRAIN_COHORT) & ~is_aug)
    sgkf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    rf_w = []
    for a, _ in sgkf.split(np.zeros(len(wtr)), y3[wtr], groups[wtr]):
        m = rf(); m.fit(Xw[wtr[a]], y3[wtr[a]]); rf_w.append(m)
    print("A0 window-level forest fitted", flush=True)
    P_A0 = np.stack([np.mean([m.predict_proba(Xw[ix]) for m in rf_w], axis=0).mean(axis=0)
                     for ix in idxs])

    print("fitting subject-level arms...", flush=True)
    P = {
        "A0_window_RF_spectral": P_A0,
        "A1_subject_RF_spectral": flat_subject(rf, Xsub)(Xsub),
        "A2_subject_RF_concepts": flat_subject(rf, Csub)(Csub),
        "A3_subject_logistic_spectral": flat_subject(lr, Xsub)(Xsub),
        "A4_subject_logistic_concepts": flat_subject(lr, Csub)(Csub),
        "A5_hier_logistic_concepts": hierarchical_subject(lr, Csub)(Csub),
    }

    # A6: the manuscript's model.
    def sc():
        return SignConstrainedLogistic(dirs, C=0.5)
    f1 = [sc().fit(Csub[tr][a], y_l1[tr][a]) for a, _ in tr_folds]
    f2 = [sc().fit(Csub[tr & imp][a], y_l2[tr & imp][a]) for a, _ in imp_folds]
    p1 = np.mean([m.predict_proba(Csub)[:, 1] for m in f1], axis=0)
    p2 = np.mean([m.predict_proba(Csub)[:, 1] for m in f2], axis=0)
    P["A6_hier_signconstrained_concepts"] = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
    print("all arms fitted\n", flush=True)

    LABELS = {
        "A0_window_RF_spectral": "window-level RF, 123 spectral  (incumbent)",
        "A1_subject_RF_spectral": "subject-level RF, 123 spectral",
        "A2_subject_RF_concepts": "subject-level RF, 14 concepts",
        "A3_subject_logistic_spectral": "subject-level logistic, 123 spectral",
        "A4_subject_logistic_concepts": "subject-level logistic, 14 concepts",
        "A5_hier_logistic_concepts": "hierarchical logistic, 14 concepts",
        "A6_hier_signconstrained_concepts": "hierarchical SIGN-CONSTRAINED  (the model)",
    }
    report = {"note": "R3 cost-of-interpretability ladder; all arms CAUEEG-trained, "
                      "identical folds and subjects, nothing tuned on the blind sites.",
              "arms": LABELS, "sites": {}}

    for site in sites:
        m = sub_coh == site
        for t in TARGETS:
            mt, yt = target_subset(sub_y3[m], t)
            if mt.sum() < 16 or len(np.unique(yt)) < 2:
                continue
            key = f"{site}|{t}"
            print("=" * 84)
            print(f"{key}  (n={int(mt.sum())})")
            ref = pairwise_score(P["A0_window_RF_spectral"][m][mt], t)
            rows = {}
            for arm, Pa in P.items():
                s = pairwise_score(Pa[m][mt], t)
                a, lo, hi = auc_ci(yt, s)
                d, dlo, dhi, dp = (None, None, None, None) if arm.startswith("A0") else \
                    paired_bootstrap(yt, s, ref)
                rows[arm] = {"auc": a, "ci95": [lo, hi], "delta_vs_A0": d,
                             "delta_ci95": [dlo, dhi], "p": dp}
                tail = ("" if d is None else
                        f"   vs A0 {d:+.3f} [{fmt(dlo)},{fmt(dhi)}] p={fmt(dp)}")
                print(f"  {LABELS[arm]:44s} {fmt(a)} [{fmt(lo)}-{fmt(hi)}]{tail}")

            # the decomposition the paper needs
            def g(k):
                return rows[k]["auc"]
            steps = {
                "training_unit_A0_minus_A1": (g("A0_window_RF_spectral")
                                              - g("A1_subject_RF_spectral")),
                "bottleneck_A1_minus_A2": (g("A1_subject_RF_spectral")
                                           - g("A2_subject_RF_concepts")),
                "linearity_A2_minus_A4": (g("A2_subject_RF_concepts")
                                          - g("A4_subject_logistic_concepts")),
                "hierarchy_A4_minus_A5": (g("A4_subject_logistic_concepts")
                                          - g("A5_hier_logistic_concepts")),
                "sign_constraints_A5_minus_A6": (g("A5_hier_logistic_concepts")
                                                 - g("A6_hier_signconstrained_concepts")),
                "dimensionality_A3_minus_A4": (g("A3_subject_logistic_spectral")
                                               - g("A4_subject_logistic_concepts")),
                "total_A0_minus_A6": (g("A0_window_RF_spectral")
                                      - g("A6_hier_signconstrained_concepts")),
            }
            print("\n  decomposition of the reported deficit (positive = costs AUC):")
            for k, v in steps.items():
                print(f"    {k:34s} {v:+.3f}")
            report["sites"][key] = {"n": int(mt.sum()), "arms": rows,
                                    "decomposition": steps}
            print(flush=True)

    from src.utils.provenance import write_result
    dest = write_result(OUT / "cost_decomposition.json", report, script=Path(__file__).name)
    print("=" * 84)
    print(f"wrote {dest}")
    print("\nHOW TO READ THIS. `total_A0_minus_A6` should reproduce the manuscript's 0.109 at "
          "OpenNeuro.\nThe share attributable to interpretability is bottleneck + linearity + "
          "hierarchy +\nsign_constraints; the share attributable to the training-unit protocol "
          "difference is\n`training_unit_A0_minus_A1`, and that share is NOT a cost of being "
          "interpretable.")


if __name__ == "__main__":
    main()
