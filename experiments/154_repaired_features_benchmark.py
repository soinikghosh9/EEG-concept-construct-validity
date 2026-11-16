"""Does repairing the two inexpressible concepts change what the classifier does?

The battery showed that `posterior_maximality` and `ap_gradient_alpha`, both named after
inter-electrode amplitude, are computed from relative band power and are therefore exactly
invariant to the amplitude they are named after. Recomputing them from absolute band power under
global scaling recovers criterion validity (rho against MMSE +0.379 and +0.492, against -0.147 and
+0.272 for the relative forms). What was never tested is the question a reader asks immediately:
what does the repair do to the decision?

The comparison has to be made inside one scaling arm. The relative forms live in the per-channel
arm and the repaired forms only exist under global scaling, so comparing across arms would confound
the estimator change with the pipeline change. This script therefore builds the matched corpus
ONCE under global scaling -- same cohorts, same 235 s truncation, same 24-window quota as the
reported corpus -- and fits the same sign-constrained head three times on it:

    relative     the fourteen concepts as reported
    repaired2    the two recovered constructs substituted in
    repaired3    all three substituted, including the one the battery did NOT recover

The third arm is the honest control: left-right asymmetry reached only +0.107 as an amplitude
measurement and remained unresolvable, so if repaired3 tracks repaired2 the gain is not coming
from "amplitude" as a blanket principle.

Sign constraints for the repaired concepts are taken from the same clinical statements that named
them, not from the observed correlations. Healthy alpha is posterior-maximal and the
anterior-posterior gradient is lost in disease, so in both cases a HIGHER amplitude ratio is the
healthier direction and the constraint is negative with respect to impairment. That the relative
forms carried the opposite constraint is not an inconsistency; it is the finding.

Reports paired deltas against the relative arm with the same subject-clustered bootstrap, seed and
resample count as 134_paired_tests.py, plus the two one-sided tests of 153_benchmark_tost.py, so
the numbers drop straight into the same tables.

    python experiments/154_repaired_features_benchmark.py
Out: outputs/results/audit/repaired_features_benchmark.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "outputs" / "results" / "audit"
FEAT_ROOT = ROOT / "outputs" / "harmonised_features"
N_BOOT, SEED = 10000, 42
MARGINS = (0.05, 0.075, 0.10)
ALPHA = 0.05

# the reported corpus, reproduced exactly -- see the provenance block of
# extract_harmonised_channel_clean.json
COHORTS = "CAUEEG,ds004504_raw,GENEEG,MCI_Dataset,AlzEEG,MISP,P-ADIC"
TRUNCATE = "CAUEEG:235,ds004504_raw:235,GENEEG:235,MISP:235,P-ADIC:235,MCI_Dataset:50"
FIXED_WINDOWS = 24
TAG = "global_clean"

SITE_TARGET = {"ds004504_raw": "CN_vs_dementia", "GENEEG": "CN_vs_MCI"}


def ensure_features(tag, force=False):
    d = FEAT_ROOT / tag
    need = (not (d / "window_concepts.npy").exists()
            or not (d / "window_amp_concepts.npy").exists())
    if force or need:
        cmd = [sys.executable, str(ROOT / "experiments" / "127_extract_harmonised.py"),
               "--mode", "global", "--tag", tag, "--cohorts", COHORTS,
               "--truncate", TRUNCATE, "--fixed-windows", str(FIXED_WINDOWS)]
        print("extracting matched corpus under global scaling:", flush=True)
        print("  " + " ".join(cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit("extraction failed with code %d" % r.returncode)
        print("extraction done in %.1f min" % ((time.time() - t0) / 60), flush=True)
    else:
        print("reusing existing features in %s" % d, flush=True)
    return d


def paired_draws(y, a, b, n_boot=N_BOOT, seed=SEED):
    """Subject-clustered paired bootstrap of AUC(a) - AUC(b), with TOST at each margin."""
    from _protocol_v2 import safe_auc
    y = np.asarray(y)
    a0, b0 = safe_auc(y, a), safe_auc(y, b)
    if a0 is None or b0 is None:
        return None
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
    two_sided = float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))
    lo90, hi90 = float(np.percentile(d, 100 * ALPHA)), float(np.percentile(d, 100 * (1 - ALPHA)))
    tost = {}
    for m in MARGINS:
        tost["%g" % m] = {
            "ci90": [lo90, hi90],
            "p_tost": max(float((d <= -m).mean()), float((d >= m).mean())),
            "equivalent": bool(lo90 > -m and hi90 < m),
        }
    return {"auc_a": a0, "auc_b": b0, "delta": a0 - b0,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": max(two_sided, 1.0 / max(len(d), 1)),
            "tost": tost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    feat = ensure_features(TAG, args.force_extract)

    from sklearn.model_selection import StratifiedKFold

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)
    from src.utils.provenance import write_result
    from _protocol_v2 import CN, DEM, MCI, safe_auc

    LAB = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI, "AD": DEM, "A": DEM,
           "FTD": DEM, "F": DEM}
    # direction with respect to IMPAIRMENT, specified from the clinical statement that names the
    # concept, exactly as for the reported model
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
           "artefact_proxy": 0, "log_n_windows": 0}
    # healthy alpha is posterior-maximal, and the anterior-posterior gradient is lost in disease:
    # a HIGHER amplitude ratio is the healthier direction in both cases
    AMP_DIR = {"amp_posterior_maximality": -1, "amp_ap_gradient_alpha": -1,
               "amp_asymmetry_alpha": 0}
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")

    C_win = np.load(feat / "window_concepts.npy")
    A_win = np.load(feat / "window_amp_concepts.npy")
    meta = np.load(feat / "meta.npz", allow_pickle=False)
    g, lb, co = meta["groups"], meta["labels"], meta["cohort"]
    subs = np.unique(g)
    idxs = [np.flatnonzero(g == s) for s in subs]
    coh = np.array([co[i[0]] for i in idxs])
    y3 = np.array([LAB[lb[i[0]]] for i in idxs])
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    # aggregate_subject maps [n_win, 15] -> 17-d, appending the two declared-but-unused
    # quantities. The amplitude array is 3-wide and must not go through it; the aggregation it
    # needs is the same median over windows, nothing more.
    with np.errstate(all="ignore"):
        A = np.stack([np.nanmedian(A_win[i], axis=0) for i in idxs])
    tr = coh == "CAUEEG"
    C, _ = impute_from_training(C, tr)
    amp_fill = np.nanmedian(A[tr], axis=0)
    A = np.where(np.isfinite(A), A, amp_fill)
    print("%d subjects | train %d | concepts %s amp %s"
          % (len(subs), tr.sum(), C.shape, A.shape), flush=True)

    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    base_names = [CONCEPT_NAMES[i] for i in cols]
    AMP_ORDER = ["amp_posterior_maximality", "amp_ap_gradient_alpha", "amp_asymmetry_alpha"]
    SUB = {"posterior_maximality": 0, "ap_gradient_alpha": 1, "asymmetry_alpha": 2}

    def build(which):
        """Return (X, names, dirs) with the named substitutions applied."""
        X = C[:, cols].copy()
        names = list(base_names)
        for rel, k in SUB.items():
            if rel not in which:
                continue
            j = names.index(rel)
            X[:, j] = A[:, k]
            names[j] = AMP_ORDER[k]
        # not AMP_DIR.get(n, DIR[n]): the default expression is evaluated eagerly and DIR has no
        # entry for the amplitude names
        dirs = [AMP_DIR[n] if n in AMP_DIR else DIR[n] for n in names]
        return X, names, dirs

    ARMS = {
        "relative": (),
        "repaired2": ("posterior_maximality", "ap_gradient_alpha"),
        "repaired3": ("posterior_maximality", "ap_gradient_alpha", "asymmetry_alpha"),
    }

    y1 = (y3 != CN).astype(int)
    preds = {}
    report = {"tag": TAG, "n_boot": N_BOOT, "margins": list(MARGINS),
              "n_subjects": int(len(subs)), "n_train": int(tr.sum()),
              "arms": {}, "sites": {}}

    for arm, which in ARMS.items():
        X, names, dirs = build(which)
        fits = [SignConstrainedLogistic(dirs, C=0.5).fit(X[tr][a], y1[tr][a])
                for a, _ in StratifiedKFold(5, shuffle=True,
                                            random_state=42).split(X[tr], y3[tr])]
        preds[arm] = {"X": X, "fits": fits}
        report["arms"][arm] = {"concepts": names, "substituted": list(which),
                               "directions": dirs}
        print("  fitted %s: %d substitution(s)" % (arm, len(which)), flush=True)

    for site, target in SITE_TARGET.items():
        m = coh == site
        if m.sum() < 16:
            continue
        yt = y1[m]
        if len(np.unique(yt)) < 2:
            continue
        label = "%s|%s" % ("OpenNeuro" if site.startswith("ds004504") else site, target)
        scores = {}
        for arm in ARMS:
            scores[arm] = np.mean([f.predict_proba(preds[arm]["X"][m])[:, 1]
                                   for f in preds[arm]["fits"]], axis=0)
        srep = {"n": int(m.sum()),
                "auc": dict((arm, safe_auc(yt, s)) for arm, s in scores.items()),
                "vs_relative": {}}
        print("\n%s  (n=%d)" % (label, int(m.sum())), flush=True)
        for arm in ("repaired2", "repaired3"):
            r = paired_draws(yt, scores[arm], scores["relative"])
            if r is None:
                continue
            srep["vs_relative"][arm] = r
            cells = []
            for mg in MARGINS:
                t = r["tost"]["%g" % mg]
                cells.append(("EQUIV" if t["equivalent"] else "  --")
                             + " %.3f" % t["p_tost"])
            print("  %-10s AUC %.4f vs %.4f  delta %+.4f  p=%.4f   %s"
                  % (arm, r["auc_a"], r["auc_b"], r["delta"], r["p"], "  ".join(cells)),
                  flush=True)
        report["sites"][label] = srep

    dest = write_result(OUT / "repaired_features_benchmark.json", report,
                        script=Path(__file__).name)
    print("\nwrote %s" % dest, flush=True)


if __name__ == "__main__":
    main()
