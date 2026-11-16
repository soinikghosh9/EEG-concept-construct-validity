"""Does the benchmark survive harmonised preprocessing?

The question this answers first: the random forest reached 0.850 at the blind dementia site on the
old pipeline. That pipeline left roughly half the training cohort's variance as ocular artefact,
removed a differing amount of it by diagnosis, and applied a Surface Laplacian to the test cohort
but not to the training cohort. If 0.850 was riding any of that, it should move now.

Same protocol as `98_benchmark.py` and `119_cost_decomposition.py` so the numbers are comparable:
train on CAUEEG only, 5 folds grouped by subject, score the blind sites with the restricted
pairwise posterior, paired subject bootstrap for the deltas. The only change is the input.

Run per scaling arm. `--mode global` is the arm that preserves inter-channel amplitude, which the
topographic concepts need and which a graph network would exploit.

    python experiments/128_benchmark_harmonised.py --mode channel
    python experiments/128_benchmark_harmonised.py --mode global
Out: outputs/results/audit/benchmark_harmonised_<mode>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, TARGETS, auc_ci, pairwise_score, safe_auc, target_subset  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEED, N_BOOT, FOLDS = 42, 5000, 5

#: Cohorts label their groups differently -- CAUEEG uses AD/CN/FTD/MCI, ds004504 uses single
#: letters A/C/F, GENEEG uses CONTROL/MCI. Mapping them by hand and asserting the resulting counts
#: against the known composition is the only safe way to do this; a silent mismatch here would
#: relabel a whole cohort with nothing downstream to reveal it.
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN,
          "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}
EXPECTED = {("CAUEEG", CN): 459, ("CAUEEG", MCI): 417, ("CAUEEG", DEM): 244,
            ("ds004504_raw", CN): 29, ("ds004504_raw", DEM): 59,
            ("GENEEG", CN): 35, ("GENEEG", MCI): 28}

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
    return (a0 - b0, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["channel", "global"], required=True)
    ap.add_argument("--tag", default=None,
                    help="feature table to read, if not the one named by --mode. Lets an "
                         "alternative corpus (e.g. the de-confounded one) be analysed "
                         "without disturbing the table other results were computed on.")
    args = ap.parse_args()
    tag = args.tag or args.mode

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

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
    print(f"mode={args.mode} | {len(groups):,} windows, {len(np.unique(groups)):,} subjects")

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    raw_lab = np.array([labels[i[0]] for i in idxs])
    unknown = sorted(set(raw_lab) - set(LABEL3))
    if unknown:
        raise SystemExit(f"unmapped group labels: {unknown}")
    y3 = np.array([LABEL3[x] for x in raw_lab])

    # The vocabulary check above (unmapped labels) is the one that catches a relabelled corpus and
    # stays fatal. A changed COUNT is a different thing: the de-confounded corpus drops subjects
    # who cannot supply the fixed window quota, by design, with per-class retention reported at
    # extraction. Refusing on that would block the corpus this project moved to, so it is a note.
    drift = []
    for (coh_name, cls), n_exp in EXPECTED.items():
        got = int(((sub_coh == coh_name) & (y3 == cls)).sum())
        if got != n_exp:
            drift.append(f"{coh_name}/class{cls}: {got} (reference corpus had {n_exp})")
    print("label vocabulary verified")
    for line in drift:
        print(f"  [note] {line}")

    C_sub = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    tr = sub_coh == "CAUEEG"
    C_sub, imputed = impute_from_training(C_sub, tr)
    if imputed.any():
        print(f"  imputed {int(imputed.sum())} missing concept values from the training median")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    dirs = [CLINICAL_DIRECTION[CONCEPT_NAMES[i]] for i in cols]
    Csub = C_sub[:, cols]

    imp = y3 != CN
    y_l1 = (y3 != CN).astype(int)
    y_l2 = (y3 == DEM).astype(int)
    w_tr = np.flatnonzero(np.isin(groups, subs[tr]))
    y3_win = np.repeat(y3, [len(i) for i in idxs])[np.argsort(np.concatenate(idxs))]

    def spectral_models():
        return {
            "random forest": RandomForestClassifier(n_estimators=400, n_jobs=6,
                                                    random_state=SEED, class_weight="balanced"),
            "gradient boosting": GradientBoostingClassifier(random_state=SEED),
            "SVM": make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", probability=True, class_weight="balanced",
                                     random_state=SEED)),
            "logistic": make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000,
                                                         class_weight="balanced")),
        }

    # ---- window-level spectral baselines, trained on CAUEEG ---------------------------
    sgkf = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(w_tr)), y3_win[w_tr], groups[w_tr]))
    P = {}
    for name, proto in spectral_models().items():
        from sklearn.base import clone
        fits = []
        for f, (a, _) in enumerate(splits):
            idx = w_tr[a]
            if name == "SVM" and len(idx) > 30000:
                idx = np.random.default_rng(SEED).choice(idx, 30000, replace=False)
            m = clone(proto); m.fit(S_win[idx], y3_win[idx]); fits.append(m)
        classes = fits[0].classes_
        Pw = np.mean([m.predict_proba(S_win) for m in fits], axis=0)
        full = np.zeros((len(S_win), 3))
        for j, c in enumerate(classes):
            full[:, int(c)] = Pw[:, j]
        P[name] = np.stack([full[i].mean(axis=0) for i in idxs])
        print(f"  fitted {name}", flush=True)

    # ---- concept bottleneck, subject level ---------------------------------------------
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    f1 = [SignConstrainedLogistic(dirs, C=0.5).fit(Csub[tr][a], y_l1[tr][a])
          for a, _ in skf.split(Csub[tr], y3[tr])]
    f2 = [SignConstrainedLogistic(dirs, C=0.5).fit(Csub[tr & imp][a], y_l2[tr & imp][a])
          for a, _ in skf.split(Csub[tr & imp], y_l2[tr & imp])]
    p1 = np.mean([m.predict_proba(Csub)[:, 1] for m in f1], axis=0)
    p2 = np.mean([m.predict_proba(Csub)[:, 1] for m in f2], axis=0)
    P["concept bottleneck"] = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
    viol = [CONCEPT_NAMES[cols[i]] for i in range(len(cols))
            if dirs[i] != 0 and np.sign(np.mean([m.coef_standardised_[i] for m in f1]))
            not in (0, dirs[i])]
    print(f"  fitted concept bottleneck | sign violations: {len(viol)}")

    #: Published values on the OLD pipeline, for the comparison this script exists to make.
    OLD = {"OpenNeuro|CN_vs_dementia": {"random forest": 0.850, "gradient boosting": 0.821,
                                        "SVM": 0.806, "logistic": 0.756,
                                        "concept bottleneck": 0.741},
           "GENEEG|CN_vs_MCI": {"random forest": 0.616, "gradient boosting": 0.593,
                                "SVM": 0.553, "logistic": 0.417,
                                "concept bottleneck": 0.579}}

    report = {"mode": args.mode, "tag": tag, "n_windows": int(len(groups)),
              "n_subjects": int(len(subs)), "sign_violations": viol, "sites": {}}

    # Subject-level posteriors, saved so the deep and classical arms can be compared with a
    # paired test on identical subjects rather than by eyeballing two overlapping intervals.
    pdir = OUT / "predictions" / tag
    pdir.mkdir(parents=True, exist_ok=True)
    for site in ("ds004504_raw", "GENEEG"):
        m = sub_coh == site
        for name, Pa in P.items():
            np.savez_compressed(pdir / f"{name.replace(' ', '_')}__{site}.npz",
                                subjects=subs[m], y3=y3[m], posterior=Pa[m])
        for t in TARGETS:
            mt, yt = target_subset(y3[m], t)
            if mt.sum() < 16 or len(np.unique(yt)) < 2:
                continue
            key = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{t}"
            print(f"\n{'='*74}\n{key}  (n={int(mt.sum())})")
            print(f"  {'model':22s} {'AUC':>7s} {'95% CI':>16s} {'old':>7s} {'change':>8s}")
            rows = {}
            ref = pairwise_score(P["random forest"][m][mt], t)
            for name, Pa in P.items():
                s = pairwise_score(Pa[m][mt], t)
                a, lo, hi = auc_ci(yt, s)
                old = OLD.get(key, {}).get(name)
                delta = (a - old) if (a is not None and old is not None) else None
                dd, dlo, dhi, dp = ((None,) * 4 if name == "random forest"
                                    else paired_bootstrap(yt, s, ref))
                rows[name] = {"auc": a, "ci95": [lo, hi], "old_pipeline": old,
                              "change_vs_old": delta, "vs_rf_delta": dd,
                              "vs_rf_ci": [dlo, dhi], "vs_rf_p": dp}
                print(f"  {name:22s} {fmt(a):>7s} [{fmt(lo)}-{fmt(hi)}] "
                      f"{fmt(old):>7s} {(f'{delta:+.3f}' if delta is not None else '   -'):>8s}")
            report["sites"][key] = {"n": int(mt.sum()), "models": rows}

    dest = write_result(OUT / f"benchmark_harmonised_{tag}.json", report,
                        script="128_benchmark_harmonised.py")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
