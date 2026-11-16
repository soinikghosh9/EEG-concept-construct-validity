"""Sensitivity of the headline benchmark to the sign constraint on `posterior_maximality`.

RETAINED AS A SENSITIVITY ANALYSIS; the premise in `164` that the constraint was wrong is false --
see that file. This script remains useful for two reasons. It shows the headline numbers do not
depend on that constraint (dementia site 0.829 -> 0.826, p=0.888; prodromal 0.560 -> 0.571,
p=0.388), and it reproduces the concept-bottleneck arm of `128` from scratch, matching the saved
artefact to three decimals at both sites, which is an independent check that the benchmark's
concept arm is reproducible from the feature tables alone.

No claim from this script appears in the manuscript.

    python experiments/165_benchmark_sign_corrected.py
Out: outputs/results/audit/benchmark_sign_corrected.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()

    import json
    import importlib.util
    from sklearn.model_selection import StratifiedKFold

    spec = importlib.util.spec_from_file_location(
        "b128", ROOT / "experiments" / "128_benchmark_harmonised.py")
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)
    from src.utils.provenance import write_result

    tag = args.tag
    d = b.FEAT / tag
    b.check_table(d)
    C_win = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([b.LABEL3[x] for x in [labels[i[0]] for i in idxs]])

    C_sub = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    tr = sub_coh == "CAUEEG"
    C_sub, _ = impute_from_training(C_sub, tr)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in b.REPORT_ONLY]
    Csub = C_sub[:, cols]
    names = [CONCEPT_NAMES[i] for i in cols]

    published = [b.CLINICAL_DIRECTION[n] for n in names]
    corrected = [(-1 if n == "posterior_maximality" else b.CLINICAL_DIRECTION[n]) for n in names]

    imp = y3 != b.CN
    y_l1, y_l2 = (y3 != b.CN).astype(int), (y3 == b.DEM).astype(int)

    def hierarchy(dirs):
        skf = StratifiedKFold(b.FOLDS, shuffle=True, random_state=b.SEED)
        f1 = [SignConstrainedLogistic(dirs, C=0.5).fit(Csub[tr][a], y_l1[tr][a])
              for a, _ in skf.split(Csub[tr], y3[tr])]
        f2 = [SignConstrainedLogistic(dirs, C=0.5).fit(Csub[tr & imp][a], y_l2[tr & imp][a])
              for a, _ in skf.split(Csub[tr & imp], y_l2[tr & imp])]
        p1 = np.mean([m.predict_proba(Csub)[:, 1] for m in f1], axis=0)
        p2 = np.mean([m.predict_proba(Csub)[:, 1] for m in f2], axis=0)
        j = names.index("posterior_maximality")
        w = float(np.mean([m.coef_standardised_[j] for m in f1]))
        return np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2]), w

    Ppub, wpub = hierarchy(published)
    Pcor, wcor = hierarchy(corrected)
    print("posterior_maximality level-1 weight: published %+.4f   corrected %+.4f" % (wpub, wcor))

    ref = json.load(open(OUT / ("benchmark_harmonised_%s.json" % tag)))["sites"]
    report = {"tag": tag, "weight_level1": {"published": wpub, "corrected": wcor}, "sites": {}}

    for site in ("ds004504_raw", "GENEEG"):
        m = sub_coh == site
        for t in b.TARGETS:
            mt, yt = b.target_subset(y3[m], t)
            if mt.sum() < 16 or len(np.unique(yt)) < 2:
                continue
            key = "%s|%s" % ("OpenNeuro" if site.startswith("ds004504") else site, t)
            sp = b.pairwise_score(Ppub[m][mt], t)
            sc = b.pairwise_score(Pcor[m][mt], t)
            dd, dlo, dhi, dp = b.paired_bootstrap(yt, sp, sc)
            ap_, _, _ = b.auc_ci(yt, sp)
            ac_, _, _ = b.auc_ci(yt, sc)
            pub_ref = ref.get(key, {}).get("models", {}).get("concept bottleneck", {}).get("auc")
            rf = ref.get(key, {}).get("models", {}).get("random forest", {}).get("auc")
            report["sites"][key] = {"n": int(mt.sum()), "auc_published": ap_,
                                    "auc_corrected": ac_, "delta": dd, "ci95": [dlo, dhi],
                                    "p": dp, "benchmark_artefact_auc": pub_ref,
                                    "random_forest_auc": rf}
            print("%-26s n=%-3d published %.3f  corrected %.3f  delta %+.3f [%+.3f,%+.3f] p=%.3f"
                  "   (artefact %s, RF %s)"
                  % (key, mt.sum(), ap_, ac_, dd, dlo, dhi, dp,
                     b.fmt(pub_ref), b.fmt(rf)), flush=True)

    dest = write_result(OUT / "benchmark_sign_corrected.json", report, script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
