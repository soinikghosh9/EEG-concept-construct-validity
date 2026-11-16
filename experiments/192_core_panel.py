"""Is the reported panel reducible without changing the model?

Five concepts carry a constrained coefficient of exactly zero on every task: the sign
constraint is a box at zero, so a concept whose data relationship runs against the direction
its clinical name asserts lands on that boundary and cannot enter the decision at all. That
is a property of the measurement, not a selection on performance -- nothing here is chosen by
comparing blind-site AUC, which is the failure this project exists to document.

If those coefficients are zero in every fold, the constrained optimum for the remaining
concepts lies in the subspace where they are zero, so refitting without them should return
the identical model. This script tests that rather than assuming it: it reports the per-fold
zero pattern, refits on the reduced panel, and compares external scores subject by subject.

    python experiments/192_core_panel.py
Out: outputs/results/audit/core_panel.json
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
SEED = 42


def _load_157():
    spec = importlib.util.spec_from_file_location(
        "task_battery", ROOT / "experiments" / "157_task_battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    X = T["X_concept"]
    report = {"tag": args.tag, "concepts_full": names, "tasks": {}}

    # ---------------------------------------------------------------- per-fold zero pattern
    zero_counts = {n: 0 for n in names}
    total_fits = 0
    per_task_nonzero = {n: 0 for n in names}
    fits_by_task = {}

    for task, (pos, neg, train_coh, eval_coh) in tb.TASKS.items():
        if eval_coh is None:
            continue
        keep = np.isin(T["cls"], pos + neg)
        y = np.isin(T["cls"], pos).astype(int)
        tr = keep & (T["cohort"] == train_coh)
        te = keep & (T["cohort"] == eval_coh)
        if tr.sum() < 40 or te.sum() < 10 or len(np.unique(y[te])) < 2:
            continue
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr]))
        d = [DIR[n] for n in names]
        fits = [SignConstrainedLogistic(d, C=0.5).fit(X[tr][a], y[tr][a]) for a, _ in folds]
        W = np.stack([np.asarray(f.coef_).ravel() for f in fits])
        for i, n in enumerate(names):
            zero_counts[n] += int((np.abs(W[:, i]) < 1e-8).sum())
            if np.abs(W[:, i]).max() > 1e-8:
                per_task_nonzero[n] += 1
        total_fits += len(fits)
        fits_by_task[task] = (folds, tr, te, y, fits)

    print(f"Per-fold zero pattern across {total_fits} fits "
          f"({len(fits_by_task)} tasks x 5 folds)")
    print(f"{'concept':22s} {'zero in':>10s} {'tasks used':>11s}")
    never = []
    for n in names:
        print(f"{n:22s} {zero_counts[n]:4d}/{total_fits:<5d} {per_task_nonzero[n]:11d}")
        if zero_counts[n] == total_fits:
            never.append(n)
    print(f"\nzero in EVERY fit ({len(never)}): {never}")
    report["zero_in_every_fit"] = never
    report["n_fits"] = total_fits
    report["zero_counts"] = zero_counts

    core = [n for n in names if n not in never]
    cols = [i for i, n in enumerate(names) if n in core]
    print(f"core panel ({len(core)}): {core}\n")
    report["core_panel"] = core

    # ---------------------------------------------------------------- refit on the core panel
    print(f"{'task':18s} {'n':>4s} | {'full (14)':>10s} {'core':>8s} {'delta':>8s} "
          f"{'max |dp|':>9s} {'identical':>10s}")
    print("-" * 78)
    all_ident = True
    for task, (folds, tr, te, y, fits_full) in fits_by_task.items():
        d = [DIR[names[i]] for i in cols]
        fits_core = [SignConstrainedLogistic(d, C=0.5).fit(X[tr][:, cols][a], y[tr][a])
                     for a, _ in folds]
        s_full = np.mean([f.predict_proba(X[te])[:, 1] for f in fits_full], axis=0)
        s_core = np.mean([f.predict_proba(X[te][:, cols])[:, 1] for f in fits_core], axis=0)
        a_full, a_core = safe_auc(y[te], s_full), safe_auc(y[te], s_core)
        dmax = float(np.abs(s_full - s_core).max())
        # L-BFGS-B converges to a tolerance, so 'identical' means within it, not bitwise
        ident = dmax < 1e-4
        all_ident &= ident
        print(f"{task:18s} {te.sum():4d} | {a_full:10.4f} {a_core:8.4f} "
              f"{a_core - a_full:+8.4f} {dmax:9.2e} {str(ident):>10s}")
        report["tasks"][task] = {"n_eval": int(te.sum()), "auc_full": float(a_full),
                                "auc_core": float(a_core),
                                "delta": float(a_core - a_full),
                                "max_abs_score_difference": dmax,
                                "identical": bool(ident)}

    report["all_tasks_identical"] = bool(all_ident)
    print(f"\nEvery task identical to within 1e-4 (optimiser tolerance): {all_ident}")
    if all_ident:
        print("The reduced panel is not an approximation of the reported model. It is the")
        print("same model, stated without the terms whose coefficient the constraint fixes")
        print("at zero. No reported figure changes.")

    write_result(OUT / "core_panel.json", report, script=__file__,
                 extra={"description":
                        "Whether the reported concept panel reduces without changing the "
                        "fitted model: per-fold zero pattern under the sign constraint, and "
                        "subject-level score comparison after refitting on the core panel."})
    print(f"\nwrote {OUT / 'core_panel.json'}")


if __name__ == "__main__":
    main()
