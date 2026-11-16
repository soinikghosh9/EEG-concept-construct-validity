"""Which concepts carry the decision, and what does each one buy?

The manuscript reports contribution shares for a model that no longer exists -- the pre-repair
fifteen-concept version on the previous pipeline. Nothing is reported for the model it now
proposes. Three complementary views are needed, because each is misleading alone:

  * **leave-one-out** -- retrain without each concept and measure what discrimination is lost.
    Honest about redundancy: a concept that duplicates another shows no loss, correctly, where a
    coefficient would have credited it anyway.
  * **contribution share** -- mean absolute contribution to the log-odds at the site where the
    decision is made. Answers "what drove this decision" rather than "what would be missed".
  * **single-concept discrimination** -- what each achieves alone, which bounds how much any
    attribution can honestly assign to it.

They are reported together because they disagree in informative ways. A concept with a large
contribution share, no leave-one-out loss and poor standalone discrimination is one the model
leans on and does not need -- and with variance inflation factors near 30, that description fits
several of these.

    python experiments/138_concept_ablation.py --mode channel
Out: outputs/results/audit/concept_ablation_<mode>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, safe_auc  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
SEED, N_BOOT = 42, 5000
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}
CLINICAL_DIRECTION = {
    "pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
    "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
    "asymmetry_alpha": 0, "asymmetry_freq": 0,
    "theta_alpha_ratio": +1, "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
    "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
    "artefact_proxy": 0, "log_n_windows": 0,
}
REPORT_ONLY = ("artefact_proxy", "log_n_windows", "spectral_entropy")
SITES = {"ds004504_raw": DEM, "GENEEG": MCI}


def fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def paired(y, a, b, n_boot=N_BOOT, seed=SEED):
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
    p = float(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))
    return {"delta": a0 - b0,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": max(p, 1.0 / max(len(d), 1))}


def vif(Z):
    out = []
    for j in range(Z.shape[1]):
        y = Z[:, j]
        X = np.column_stack([np.delete(Z, j, axis=1), np.ones(len(Z))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1.0 - float(((y - X @ beta) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))
        out.append(float(1.0 / max(1.0 - r2, 1e-9)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    ap.add_argument("--tag", default=None,
                    help="feature table to read, if not the one named by --mode. Lets an "
                         "alternative corpus (e.g. the de-confounded one) be analysed "
                         "without disturbing the table other results were computed on.")
    args = ap.parse_args()
    tag = args.tag or args.mode
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)

    d = FEAT / tag

    # every array in this directory is joined by row index; a stale one
    # would silently pair features with the wrong windows
    check_table(d)

    C_win = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    C_all = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    tr_all = sub_coh == "CAUEEG"
    C_all, _ = impute_from_training(C_all, tr_all)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in REPORT_ONLY]
    names = [CONCEPT_NAMES[i] for i in cols]
    dirs = [CLINICAL_DIRECTION[n] for n in names]
    C = C_all[:, cols]
    print(f"mode={args.mode} | {len(names)} concepts in the classifier")

    report = {"mode": args.mode, "tag": tag, "concepts": names, "targets": {}}

    for site, pos in SITES.items():
        tr = tr_all & np.isin(y3, [CN, pos])
        te = (sub_coh == site) & np.isin(y3, [CN, pos])
        if te.sum() < 16:
            continue
        y_tr = (y3[tr] == pos).astype(int)
        y_te = (y3[te] == pos).astype(int)
        lbl = (f"{'OpenNeuro' if site.startswith('ds004504') else site}|"
               f"{'CN_vs_dementia' if pos == DEM else 'CN_vs_MCI'}")
        print(f"\n{'='*88}\n{lbl}   train n={int(tr.sum())}  test n={int(te.sum())}")

        def fit_predict(keep_idx, constrained):
            A, B = C[tr][:, keep_idx], C[te][:, keep_idx]
            if constrained:
                m = SignConstrainedLogistic([dirs[i] for i in keep_idx], C=0.5).fit(A, y_tr)
            else:
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000,
                                                     class_weight="balanced")).fit(A, y_tr)
            return m.predict_proba(B)[:, 1], m

        full_idx = list(range(len(names)))
        rows = {}
        for constrained in (True, False):
            arm = "sign-constrained" if constrained else "unconstrained"
            base_p, base_m = fit_predict(full_idx, constrained)
            base_auc = safe_auc(y_te, base_p)
            share = None
            if constrained:
                contrib = np.abs(base_m.contributions(C[te][:, full_idx])).mean(axis=0)
                share = contrib / (contrib.sum() + 1e-12)
            print(f"\n  {arm}: full model AUC {fmt(base_auc)}")
            print(f"  {'concept':22s} {'LOO auc':>8s} {'delta':>8s} {'p':>7s} "
                  f"{'alone':>7s} {'contrib%':>9s}")
            per = {}
            for j, nm in enumerate(names):
                keep = [k for k in full_idx if k != j]
                p_loo, _ = fit_predict(keep, constrained)
                st = paired(y_te, p_loo, base_p)
                alone = safe_auc(y_te, C[te][:, j] * (dirs[j] if dirs[j] else 1))
                per[nm] = {"loo_auc": safe_auc(y_te, p_loo),
                           "delta_vs_full": st["delta"] if st else None,
                           "ci95": st["ci95"] if st else None,
                           "p": st["p"] if st else None, "alone_auc": alone,
                           "contribution_share": float(share[j]) if share is not None else None}
                pstr = f"{st['p']:.3f}" if st else "  n/a"
                sstr = f"{100*share[j]:.1f}%" if share is not None else "   -"
                print(f"  {nm:22s} {fmt(per[nm]['loo_auc']):>8s} "
                      f"{fmt(per[nm]['delta_vs_full']):>8s} {pstr:>7s} "
                      f"{fmt(alone):>7s} {sstr:>9s}")
            rows[arm] = {"full_auc": base_auc, "per_concept": per}

        Ztr = (C[tr] - C[tr].mean(0)) / (C[tr].std(0) + 1e-9)
        v = vif(Ztr)
        rows["collinearity"] = {"condition_number": float(np.linalg.cond(Ztr)),
                                "vif": dict(zip(names, v))}
        print(f"\n  collinearity: condition number {np.linalg.cond(Ztr):.1f}, "
              f"max VIF {max(v):.1f} ({names[int(np.argmax(v))]})")
        report["targets"][lbl] = rows

    dest = write_result(OUT / f"concept_ablation_{tag}.json", report,
                        script="138_concept_ablation.py")
    print(f"\nwrote {dest}")
    print("\nA concept with a large contribution share, no leave-one-out loss and weak standalone "
          "AUC is\none the model leans on but does not need. With VIFs at this level that is "
          "expected for\nseveral of them, and no single-concept claim should be made without it.")


if __name__ == "__main__":
    main()
