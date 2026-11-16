"""How much of the decision did the artefact-dominated concepts actually carry?

`120` answered this, but on the pre-harmonisation corpus (CAUEEG 1,120, OpenNeuro 88) and writing
its JSON directly rather than through `write_result`, so the artefact carries no provenance block
and describes a corpus no other number in the manuscript is computed on. The 66.7 per cent quoted
in the results therefore could not be traced to a commit or to the analysed 1,296 subjects.

This recomputes it on the harmonised corpus. The claim concerns the WIDE-band configuration -- the
one common in the literature, before the analysis band is restricted -- so the wide arm is the
subject of the measurement and the restricted arm is reported beside it to show what the repair
bought.

Contribution is the exact per-concept term in the log-odds, |w_j z_j| normalised across concepts
and averaged over subjects at the evaluation site, not the coefficient magnitude: coefficients are
partial derivatives, and two concepts with equal coefficients contribute differently when their
spreads at the target site differ.

    python experiments/167_contribution_harmonised.py
Out: outputs/results/audit/contribution_share_harmonised.json
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
SEED, FOLDS, N_BOOT = 42, 5, 2000

#: The three the battery flagged as artefact-dominated under wide bands.
FLAGGED = ("aperiodic_exponent", "spectral_edge", "spectral_entropy")
#: Never enter any classifier, in any arm.
NEVER = ("artefact_proxy", "log_n_windows")

DIRS = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
        "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
        "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
        "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
        "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0}


def arm(tag, drop_entropy):
    """Fit the screening stage on the training cohort, score contributions at the blind site."""
    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)
    from src.data.feature_table import check_table

    d = FEAT / tag
    check_table(d)
    W = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    coh = np.array([cohort[i[0]] for i in idxs])
    lab = np.array([labels[i[0]] for i in idxs])
    y = (~np.isin(lab, ("CN", "C", "CONTROL", "Normal", "normal"))).astype(int)

    C = np.stack([aggregate_subject(W[ix]) for ix in idxs])
    tr = coh == "CAUEEG"
    C, _ = impute_from_training(C, tr)

    excl = set(NEVER) | ({"spectral_entropy"} if drop_entropy else set())
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in excl]
    names = [CONCEPT_NAMES[i] for i in cols]
    Z, dirs = C[:, cols], [DIRS[n] for n in names]

    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    fits = [SignConstrainedLogistic(dirs, C=0.5).fit(Z[tr][a], y[tr][a])
            for a, _ in skf.split(Z[tr], y[tr])]

    site = coh == "ds004504_raw"
    Zs = Z[site]

    # contribution_j(subject) = |w_j * z_j| on the standardised scale the model actually uses
    per_fit = []
    for f in fits:
        Zstd = (Zs - f.mu_) / f.sd_
        con = np.abs(Zstd * np.asarray(f.coef_standardised_).ravel())
        per_fit.append(con / np.maximum(con.sum(axis=1, keepdims=True), 1e-12))
    con = np.mean(per_fit, axis=0)                    # [n_site, n_concept]

    share = con.mean(axis=0)
    flag = [j for j, n in enumerate(names) if n in FLAGGED]
    tot = con[:, flag].sum(axis=1)

    rng = np.random.default_rng(SEED)
    boot = [tot[rng.integers(0, len(tot), len(tot))].mean() for _ in range(N_BOOT)]

    return {"tag": tag, "n_site": int(site.sum()), "n_train": int(tr.sum()),
            "concepts": names,
            "flagged_present": [names[j] for j in flag],
            "share_per_concept": {n: float(share[j]) for j, n in enumerate(names)},
            "flagged_contribution_share": float(tot.mean()),
            "flagged_ci95": [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
            "coefficient_share_flagged": float(
                sum(abs(np.mean([f.coef_standardised_[j] for f in fits])) for j in flag)
                / max(sum(abs(np.mean([f.coef_standardised_[j] for f in fits]))
                          for j in range(len(names))), 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    from src.utils.provenance import write_result

    report = {"note": "Contribution of the artefact-dominated concepts to the screening decision, "
                      "on the harmonised corpus. Supersedes contribution_share.json, which was "
                      "computed on the pre-harmonisation corpus and carried no provenance block.",
              "arms": {}}

    for tag, drop in (("channel_clean_orig", False), ("channel_clean", True)):
        try:
            r = arm(tag, drop)
        except Exception as e:                                          # noqa: BLE001
            print("  [skip] %s: %s: %s" % (tag, type(e).__name__, str(e)[:110]))
            continue
        report["arms"][tag] = r
        print("\n=== %s   n_train=%d  n_site=%d" % (tag, r["n_train"], r["n_site"]))
        print("   flagged concepts present: %s" % ", ".join(r["flagged_present"]))
        print("   contribution share %.3f  [%.3f, %.3f]   (coefficient share %.3f)"
              % (r["flagged_contribution_share"], r["flagged_ci95"][0], r["flagged_ci95"][1],
                 r["coefficient_share_flagged"]))
        top = sorted(r["share_per_concept"].items(), key=lambda kv: -kv[1])[:5]
        print("   largest contributors: %s"
              % ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in top))

    dest = write_result(OUT / "contribution_share_harmonised.json", report,
                        script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
