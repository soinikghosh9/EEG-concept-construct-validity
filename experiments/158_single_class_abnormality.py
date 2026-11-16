"""Can the tool recognise abnormality in cohorts that contain no controls?

Two cohorts contribute mild cognitive impairment and nothing else. They admit no area under the
curve, so the benchmark ignores them, and that is a waste: a triage tool meets single-class
populations constantly, and what it does with a ward full of patients and no controls is a fair
question about whether it has learned anything transferable.

Three things are asked of each cohort.

  1. **Referral rate.** At the threshold set from the training cohort's prevalence, which uses
     nothing from the target site, what fraction of an all-patient cohort is referred? A tool that
     works should refer most of them; one that refers half is not distinguishing anything.

  2. **Concept-level abnormality.** For each concept, where does the cohort sit relative to the
     training controls, in robust z units, and is the displacement in the direction the clinical
     literature specifies? This is the check that does not need a control group: the direction is
     specified in advance, so agreement is falsifiable.

  3. **Refusal.** How much of the cohort falls outside the training distribution in concept space?
     A cohort that is largely refused tells us the referral rate above is not trustworthy.

Reference is the training cohort's own MCI subjects, so "as abnormal as expected" has a concrete
meaning rather than a rhetorical one.

    python experiments/158_single_class_abnormality.py
Out: outputs/results/audit/single_class_abnormality.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "outputs" / "results" / "audit"
SEED = 42
SINGLE = ("MISP", "P-ADIC")

# direction with respect to impairment, from the clinical statement naming each concept
DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic
    from src.utils.provenance import write_result
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    names, X, coh, cls = T["names"], T["X_concept"], T["cohort"], T["cls"]
    tr = coh == "CAUEEG"
    y1 = (cls != "CN").astype(int)

    # screening model, trained exactly as reported
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y1[tr]))
    fits = [SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(X[tr][a], y1[tr][a])
            for a, _ in folds]

    def score(Z):
        return np.mean([f.predict_proba(Z)[:, 1] for f in fits], axis=0)

    p_train = float(y1[tr].mean())
    s_tr = score(X[tr])
    thr = float(np.quantile(s_tr, 1 - p_train))

    # robust reference distribution: training controls
    ctrl = tr & (cls == "CN")
    med = np.median(X[ctrl], axis=0)
    iqr = np.subtract(*np.percentile(X[ctrl], [75, 25], axis=0))
    iqr[iqr == 0] = np.nan

    # out-of-distribution guard, as reported
    mu = X[tr].mean(0)
    S = np.cov(X[tr], rowvar=False) + 1e-6 * np.eye(X.shape[1])
    Si = np.linalg.inv(S)

    def mahal(Z):
        d = Z - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, Si, d))

    ood_cut = float(np.quantile(mahal(X[tr]), 0.95))

    report = {"tag": args.tag, "screening_threshold": thr, "train_prevalence": p_train,
              "ood_cutoff_95th": ood_cut, "cohorts": {}}

    # the training cohort's own MCI subjects give "expected" a concrete value
    groups = {"CAUEEG MCI (reference)": tr & (cls == "MCI"),
              "CAUEEG CN (reference)": ctrl}
    for c in SINGLE:
        groups[c] = coh == c

    for label, m in groups.items():
        if m.sum() == 0:
            continue
        Z = X[m]
        sc = score(Z)
        z = (Z - med) / iqr
        agree = []
        per_concept = {}
        for j, n in enumerate(names):
            zj = float(np.nanmedian(z[:, j]))
            expected = DIR[n]
            ok = None if expected == 0 else bool(np.sign(zj) == expected)
            if ok is not None:
                agree.append(ok)
            per_concept[n] = {"median_z_vs_training_controls": zj,
                              "expected_direction": expected,
                              "direction_agrees": ok}
        d = mahal(Z)
        report["cohorts"][label] = {
            "n": int(m.sum()),
            "referred_fraction": float((sc >= thr).mean()),
            "median_score": float(np.median(sc)),
            "refused_fraction": float((d > ood_cut).mean()),
            "directional_agreement": (float(np.mean(agree)) if agree else None),
            "n_directional_concepts": len(agree),
            "concepts": per_concept}
        r = report["cohorts"][label]
        print("%-24s n=%3d  referred %.2f  refused %.2f  direction agrees %s"
              % (label, r["n"], r["referred_fraction"], r["refused_fraction"],
                 ("%.2f" % r["directional_agreement"]) if agree else "n/a"), flush=True)

    dest = write_result(OUT / "single_class_abnormality.json", report,
                        script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
