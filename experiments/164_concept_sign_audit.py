"""Sensitivity of each task to the sign constraint on `posterior_maximality`.

ORIGINALLY WRITTEN ON A FALSE PREMISE, AND RETAINED ONLY AS A SENSITIVITY ANALYSIS.

The premise was that the published +1 constraint contradicted the concept's definition. It does
not. In the analysed (per-channel) arm `posterior_maximality` is the mean RANK of the occipital
channels in alpha power with rank 1 the strongest, so a HIGHER value means occipital alpha is no
longer dominant, which is the abnormal direction, and +1 is correct. The estimator that runs the
other way is `amp_posterior_maximality`, log10(occipital / whole-head ABSOLUTE alpha), used only in
the global scaling arm, where higher is healthier and the constraint is correctly negative. Two
estimators share the clinical name and take opposite constraints, which the manuscript already
states; reading the wrong one is what produced the false alarm.

What the script still measures is therefore the cost of BREAKING the constraint, which is a
legitimate robustness question: how much of each task's performance rests on this one constraint?
The answer is that the three dementia-side tasks do not move at all (|delta| < 0.005, p >= 0.73)
because the concept is inert there in either direction, while the prodromal task loses 0.038.

No claim from this script appears in the manuscript.

    python experiments/164_concept_sign_audit.py
Out: outputs/results/audit/concept_sign_audit.json
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
SEED, N_BOOT = 42, 2000

# The table every downstream script uses.
PUBLISHED = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
             "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
             "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
             "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
             "aperiodic_exponent": 0, "spectral_edge": -1}

# The flipped table. NOT a correction -- see the module docstring; this exists to price how much
# each task depends on this one constraint.
FLIPPED = dict(PUBLISHED, posterior_maximality=-1)

RATIONALE = {
    "posterior_maximality":
        "mean rank of the occipital channels in alpha power, rank 1 strongest, so higher means "
        "occipital alpha is no longer dominant and +1 is the correct constraint; this arm flips "
        "it to measure how much each task rests on that one constraint",
}


def boot_delta(y, a, b, seed=SEED):
    """Paired bootstrap on the AUC difference between two score vectors on the same subjects."""
    from _protocol_v2 import safe_auc
    y = np.asarray(y)
    a0, b0 = safe_auc(y, a), safe_auc(y, b)
    if a0 is None or b0 is None:
        return None
    rng = np.random.default_rng(seed)
    a, b, n = np.asarray(a), np.asarray(b), len(y)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        da, db = safe_auc(y[i], a[i]), safe_auc(y[i], b[i])
        if da is not None and db is not None:
            d.append(da - db)
    d = np.asarray(d)
    p = float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))
    return {"auc_published": a0, "auc_corrected": b0, "delta": a0 - b0,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": max(p, 1.0 / max(len(d), 1))}


def fit_predict(X, y, tr, te, names, dirs):
    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic
    order = [dirs[n] for n in names]
    fits = [SignConstrainedLogistic(order, C=0.5).fit(X[tr][a], y[tr][a])
            for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr])]
    pr = np.mean([f.predict_proba(X[te])[:, 1] for f in fits], axis=0)
    w = np.stack([np.asarray(f.coef_standardised_).ravel() for f in fits]).mean(0)
    return pr, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()

    import json
    from src.utils.provenance import write_result
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    names, X, coh, cls = T["names"], T["X_concept"], T["cohort"], T["cls"]

    disagree = {k: {"published": PUBLISHED[k], "flipped": FLIPPED[k],
                    "rationale": RATIONALE.get(k)}
                for k in PUBLISHED if PUBLISHED[k] != FLIPPED[k]}
    print("=== CONSTRAINT FLIPPED (sensitivity analysis, not a correction)")
    for k, v in disagree.items():
        print("   %-22s published %+d, flipped to %+d" % (k, v["published"], v["flipped"]))
        print("      %s" % v["rationale"])

    report = {"tag": args.tag, "n_boot": N_BOOT, "premise_was_false": True,
              "flipped": disagree, "tasks": {}}
    print("\n=== EFFECT OF FLIPPING THE SIGN")
    for task, (pos, neg, trc, evc) in tb.TASKS.items():
        if evc is None:
            continue
        keep = np.isin(cls, pos + neg)
        y = np.isin(cls, pos).astype(int)
        tr, te = keep & (coh == trc), keep & (coh == evc)
        if tr.sum() < 40 or te.sum() < 16 or len(np.unique(y[te])) < 2:
            continue
        pa, wa = fit_predict(X, y, tr, te, names, PUBLISHED)
        pb, wb = fit_predict(X, y, tr, te, names, FLIPPED)
        r = boot_delta(y[te], pa, pb)
        if r is None:
            continue
        j = names.index("posterior_maximality")
        r["n_eval"] = int(te.sum())
        r["posterior_maximality_weight"] = {"published": float(wa[j]), "corrected": float(wb[j])}
        r["n_active_published"] = int((np.abs(wa) > 1e-8).sum())
        r["n_active_corrected"] = int((np.abs(wb) > 1e-8).sum())
        report["tasks"][task] = r
        print("   %-16s published %.3f  flipped %.3f  delta %+.3f [%+.3f,%+.3f]  p=%.3f"
              % (task, r["auc_published"], r["auc_corrected"], r["delta"],
                 r["ci95"][0], r["ci95"][1], r["p"]), flush=True)
        print("      posterior_maximality weight %+.3f -> %+.3f   active concepts %d -> %d"
              % (wa[j], wb[j], r["n_active_published"], r["n_active_corrected"]), flush=True)

    dest = write_result(OUT / "concept_sign_audit.json", report, script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
