"""What a forehead-and-ear montage costs the concept battery, measured on our own corpus.

A wearable array of four forehead and twelve around-the-ear electrodes (the EasyCog
configuration) has no electrode posterior of the ear. Six of the fifteen concepts name a
posterior or occipital quantity and cannot be computed there at all:

    pdr_frequency  pdr_prominence  pdr_variability
    posterior_maximality  ap_gradient_alpha  ap_gradient_freq

This script refits the sign-constrained head on the eight remaining classifier concepts,
under the *identical* protocol of 157_task_battery.py -- same folds, same C, same
directions, same imputation, same per-task binary heads -- and reports the paired change
at each external site. It needs no wearable data: it bounds what any forehead-and-ear
version of the tool could reach, using the corpus we already have.

Reported: per-task external AUC for the full and restricted heads, the paired
subject-clustered bootstrap of the difference, and the fitted coefficients of both.

    python experiments/190_wearable_montage_ablation.py
Out: outputs/results/audit/wearable_montage_ablation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "outputs" / "results" / "audit"
SEED, N_BOOT = 42, 10000

#: Concepts whose estimator reads a posterior or occipital electrode. On a forehead-and-ear
#: array both regions are empty, so each returns NaN rather than a wrong number -- provided
#: the coordinates handed to the extractor are honest (see 191_easycog_battery.py).
POSTERIOR_ONLY = ("pdr_frequency", "pdr_prominence", "pdr_variability",
                  "posterior_maximality", "ap_gradient_alpha", "ap_gradient_freq")


def _load_157():
    spec = importlib.util.spec_from_file_location(
        "task_battery", ROOT / "experiments" / "157_task_battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paired_boot(y, s_a, s_b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over subjects, both scores resampled together. Returns (delta, lo, hi, p)."""
    from _protocol_v2 import safe_auc
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    obs = safe_auc(y, s_a) - safe_auc(y, s_b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        va, vb = safe_auc(y[i], s_a[i]), safe_auc(y[i], s_b[i])
        if va is not None and vb is not None:
            d.append(va - vb)
    d = np.asarray(d, float)
    if d.size == 0:
        return obs, None, None, None
    # two-sided bootstrap p for "no difference"
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold

    from _protocol_v2 import safe_auc
    from src.models.v6 import SignConstrainedLogistic
    from src.utils.provenance import write_result

    tb = _load_157()
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1}

    T = tb.subject_table(args.tag)
    names = T["names"]
    keep_cols = [i for i, n in enumerate(names) if n not in POSTERIOR_ONLY]
    kept = [names[i] for i in keep_cols]
    dropped = [n for n in names if n in POSTERIOR_ONLY]
    print(f"corpus {args.tag}: {len(T['subjects'])} subjects")
    print(f"full head     {len(names):2d} concepts")
    print(f"wearable head {len(kept):2d} concepts -> {kept}")
    print(f"removed       {len(dropped):2d} -> {dropped}\n")

    report = {"tag": args.tag, "n_boot": N_BOOT, "seed": SEED,
              "concepts_full": names, "concepts_wearable": kept,
              "concepts_removed": dropped, "tasks": {}}

    print(f"{'task':18s} {'site':14s} {'n':>4s} {'pos':>4s} | {'full':>6s} {'wearable':>9s} "
          f"{'delta':>7s} {'95% CI':>18s} {'p':>7s}")
    print("-" * 100)

    for task, (pos, neg, train_coh, eval_coh) in tb.TASKS.items():
        if eval_coh is None:
            continue
        keep = np.isin(T["cls"], pos + neg)
        y = np.isin(T["cls"], pos).astype(int)
        tr = keep & (T["cohort"] == train_coh)
        te = keep & (T["cohort"] == eval_coh)
        if tr.sum() < 40 or te.sum() < 10 or len(np.unique(y[te])) < 2:
            continue

        X = T["X_concept"]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr]))

        def fit_head(cols):
            d = [DIR[names[i]] for i in cols]
            fits = [SignConstrainedLogistic(d, C=0.5).fit(X[tr][:, cols][a], y[tr][a])
                    for a, _ in folds]
            coef = np.mean([np.asarray(f.coef_standardised_).ravel() for f in fits], axis=0)
            return (lambda Z: np.mean([f.predict_proba(Z[:, cols])[:, 1] for f in fits], axis=0),
                    coef)

        pred_full, coef_full = fit_head(list(range(len(names))))
        pred_wear, coef_wear = fit_head(keep_cols)

        s_full, s_wear, yte = pred_full(X[te]), pred_wear(X[te]), y[te]
        a_full, a_wear = safe_auc(yte, s_full), safe_auc(yte, s_wear)
        delta, lo, hi, p = paired_boot(yte, s_wear, s_full)

        print(f"{task:18s} {eval_coh:14s} {te.sum():4d} {yte.sum():4d} | "
              f"{a_full:6.3f} {a_wear:9.3f} {delta:+7.3f} "
              f"[{lo:+.3f},{hi:+.3f}]".ljust(19) + f" {p:7.3f}")

        report["tasks"][task] = {
            "eval_cohort": eval_coh, "n_eval": int(te.sum()), "n_eval_pos": int(yte.sum()),
            "n_train": int(tr.sum()),
            "auc_full": float(a_full), "auc_wearable": float(a_wear),
            "delta": float(delta), "ci": [lo, hi], "p_paired": p,
            "coef_full": {n: float(coef_full[i]) for i, n in enumerate(names)},
            "coef_wearable": {n: float(coef_wear[i]) for i, n in enumerate(kept)},
        }

    print("\nStandardised coefficients, screening head (CN_vs_impaired):")
    scr = report["tasks"].get("CN_vs_impaired")
    if scr:
        print(f"  {'concept':22s} {'full':>9s} {'wearable':>10s}")
        for n in names:
            w = scr["coef_wearable"].get(n)
            ws = f"{w:+10.3f}" if w is not None else f"{'absent':>10s}"
            print(f"  {n:22s} {scr['coef_full'][n]:+9.3f} {ws}")

    write_result(OUT / "wearable_montage_ablation.json", report, script=__file__,
                 extra={"description":
                        "Cost of removing the six posterior-dependent concepts, which a "
                        "forehead-and-ear montage cannot supply, under the per-task "
                        "protocol of 157_task_battery.py."})
    print(f"\nwrote {OUT / 'wearable_montage_ablation.json'}")


if __name__ == "__main__":
    main()
