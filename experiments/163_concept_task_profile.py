"""Which concept does the work on which task, and does that change with the clinical question?

The attribution result says the concept model rests on a few concepts. It does not say whether the
same few serve every question. A screening decision, an Alzheimer's decision and a prodromal
decision are different clinical problems, and if the bottleneck is doing what it claims then the
concepts carrying them should differ in a way a neurophysiologist can recognise, or be the same in
a way that is itself informative.

For each task this reports, per concept: the fitted weight, its share of the total absolute weight,
permutation importance with a bootstrap interval, and the battery verdict. Concepts are ranked
within a task by permutation importance, and a concept is called task-specific when its importance
interval excludes zero on one task and includes zero on another.

Importance is recomputed here rather than reused so that every task uses the same number of
permutations and the same seed, which the earlier run did not guarantee across model families.

    python experiments/163_concept_task_profile.py
Out: outputs/results/audit/concept_task_profile.json
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
SEED, N_PERM = 42, 200

DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}

CLINICAL = {
    "rel_theta": "excess theta, the canonical slowing marker",
    "theta_alpha_ratio": "slowing expressed as a ratio; collinear with excess theta",
    "slowing_ratio": "graded slowing across four bands",
    "pdr_frequency": "posterior dominant rhythm frequency",
    "pdr_prominence": "whether a posterior rhythm is well formed",
    "pdr_variability": "instability of the dominant frequency",
    "asymmetry_alpha": "interhemispheric alpha asymmetry; a focal rather than diffuse sign",
    "asymmetry_freq": "interhemispheric frequency asymmetry",
    "posterior_maximality": "whether alpha is maximal posteriorly",
    "ap_gradient_alpha": "anterior-posterior alpha gradient",
    "ap_gradient_freq": "the same gradient on peak frequency",
    "rel_alpha": "reduced alpha power",
    "aperiodic_exponent": "non-oscillatory background slope",
    "spectral_edge": "spectral centre of mass",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--substitute", action="store_true",
                    help="attribute the repaired instrument instead of the relative one: "
                         "the two recovered amplitude estimators occupy their slots and the "
                         "occipital-maximality constraint flips with them")
    ap.add_argument("--validity-tag", default=None,
                    help="tag of the validity artefact to join, when it differs from the "
                         "feature tag (the global arm reuses the channel arm's battery)")
    args = ap.parse_args()

    import json
    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic
    from src.utils.provenance import write_result
    from _protocol_v2 import safe_auc
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    vtag = args.validity_tag or args.tag
    val = json.load(open(OUT / ("concept_validity_harmonised_%s.json" % vtag)))["concepts"]
    T = tb.subject_table(args.tag, substitute=args.substitute)
    names, X, coh, cls = T["names"], T["X_concept"], T["cohort"], T["cls"]
    if args.substitute:
        # Occipital maximality reverses polarity between its relative and amplitude forms;
        # the anterior-posterior gradient does not. See 157_task_battery.py.
        DIR.update(posterior_maximality=-1, ap_gradient_alpha=-1)

    report = {"tag": args.tag, "validity_tag": vtag, "n_permutations": N_PERM,
              "substituted": (sorted(tb.SUBSTITUTE) if args.substitute else []),
              "tasks": {}}
    for task, (pos, neg, trc, evc) in tb.TASKS.items():
        if evc is None:
            continue
        keep = np.isin(cls, pos + neg)
        y = np.isin(cls, pos).astype(int)
        tr, te = keep & (coh == trc), keep & (coh == evc)
        if tr.sum() < 40 or te.sum() < 16 or len(np.unique(y[te])) < 2:
            continue
        dirs = [DIR[n] for n in names]
        fits = [SignConstrainedLogistic(dirs, C=0.5).fit(X[tr][a], y[tr][a])
                for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr])]
        # coef_ is de-standardised into native clinical units (Hz, log-ratios, fractions), so its
        # magnitude tracks the unit a concept happens to be measured in and cannot be compared
        # across concepts. coef_standardised_ is the per-SD coefficient and is the comparable one;
        # share is computed from it. Both are reported so the native weight stays readable.
        W = np.stack([np.asarray(f.coef_standardised_).ravel() for f in fits])
        Wn = np.stack([np.asarray(f.coef_).ravel() for f in fits])
        w, w_native = W.mean(0), Wn.mean(0)
        tot = np.abs(w).sum() or 1.0

        def pred(Z):
            return np.mean([f.predict_proba(Z)[:, 1] for f in fits], axis=0)

        Xe, ye = X[te].copy(), y[te]
        base = safe_auc(ye, pred(Xe))
        rng = np.random.default_rng(SEED)
        rows = {}
        for j, n in enumerate(names):
            drops = []
            for _ in range(N_PERM):
                Z = Xe.copy()
                Z[:, j] = Z[rng.permutation(len(Z)), j]
                a = safe_auc(ye, pred(Z))
                if a is not None:
                    drops.append(base - a)
            d = np.asarray(drops)
            rows[n] = {"weight_standardised": float(w[j]),
                       "weight_native": float(w_native[j]),
                       "weight_share": float(abs(w[j]) / tot),
                       "active": bool(abs(w[j]) > 1e-8),
                       "importance": float(d.mean()),
                       "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       "significant": bool(np.percentile(d, 2.5) > 0),
                       "verdict": val.get(n, {}).get("verdict"),
                       "rho_mmse": val.get(n, {}).get("rho_mmse"),
                       "expected_direction": DIR[n],
                       "clinical": CLINICAL.get(n)}
        order = sorted(rows, key=lambda k: -rows[k]["importance"])
        for r, n in enumerate(order, 1):
            rows[n]["rank"] = r
        report["tasks"][task] = {"baseline_auc": base, "n_eval": int(te.sum()),
                                 "n_active": int(sum(v["active"] for v in rows.values())),
                                 "concepts": rows}
        print("\n=== %-16s AUC %.3f   %d active" % (task, base, report["tasks"][task]["n_active"]))
        for n in order[:6]:
            v = rows[n]
            print("   %d. %-22s imp %+.4f [%+.3f,%+.3f]%s  w %+7.2f (%2.0f%%)  %s"
                  % (v["rank"], n, v["importance"], v["ci95"][0], v["ci95"][1],
                     " *" if v["significant"] else "  ", v["weight_standardised"],
                     100 * v["weight_share"], v["verdict"]), flush=True)

    # which concepts change role across tasks
    tasks = list(report["tasks"])
    print("\n=== CONCEPT ROLE ACROSS TASKS (rank by importance)")
    print("   %-22s %s" % ("concept", "".join("%16s" % t.replace("CN_vs_", "") for t in tasks)))
    for n in sorted(names):
        cells = []
        for t in tasks:
            v = report["tasks"][t]["concepts"][n]
            cells.append("%2d%s" % (v["rank"], "*" if v["significant"] else " "))
        print("   %-22s %s" % (n, "".join("%16s" % c for c in cells)))

    stem = "concept_task_profile_repaired" if args.substitute else "concept_task_profile"
    dest = write_result(OUT / f"{stem}.json", report, script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
