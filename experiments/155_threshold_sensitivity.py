"""What does the operating threshold cost when the target site's prevalence is not known?

The manuscript states that no external cohort contributes to training, validation, model selection,
thresholding or calibration. Discrimination and the out-of-distribution guard honour that. The
screening threshold does not: `135_hierarchical_report.py` sets it to the quantile matching the
target site's own prevalence, computed from that site's labels. Assuming a known target prevalence
is a reasonable deployment position -- a clinic usually knows its case mix even when it does not
know any individual diagnosis -- but it is not label-free, and the difference should be measured
rather than asserted either way.

Three rules are compared on identical predictions:

    target      quantile matching the target site's own prevalence   (the reported rule; uses
                the target labels)
    source      quantile matching the TRAINING cohort's prevalence   (label-free at the target;
                what a blind deployment could actually do)
    youden      the threshold maximising sensitivity + specificity   (an oracle upper bound, to
                show what perfect per-site tuning would be worth)

`source` and `target` differ only in where the prevalence comes from, so the gap between them is
exactly the price of not knowing the target's case mix. `youden` bounds what any threshold rule
could achieve on that site and is reported so the other two can be read against a ceiling.

    python experiments/155_threshold_sensitivity.py --tag channel_clean
Out: outputs/results/audit/threshold_sensitivity_<tag>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, pairwise_score, safe_auc, target_subset  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
SITE_TARGET = {"ds004504_raw": "CN_vs_dementia", "GENEEG": "CN_vs_MCI"}
MODELS = ("concept_bottleneck", "random_forest")
TRAIN_COHORT_PREVALENCE = None   # computed below from the training feature table


def operating(y, score, thr):
    y = np.asarray(y)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec,
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "balanced_accuracy": 0.5 * (sens + spec), "threshold": float(thr)}


def youden_threshold(y, score):
    """The threshold maximising sensitivity + specificity, chosen ON the target labels."""
    best, best_thr = -1.0, float(np.median(score))
    for t in np.unique(score):
        o = operating(y, score, t)
        j = o["sensitivity"] + o["specificity"]
        if j > best:
            best, best_thr = j, float(t)
    return best_thr


def training_prevalence(tag):
    """Impaired fraction of the training cohort, from the feature table the models were fitted on."""
    feat = ROOT / "outputs" / "harmonised_features" / tag / "meta.npz"
    if not feat.exists():
        return None
    m = np.load(feat, allow_pickle=False)
    g, lb, co = m["groups"], m["labels"], m["cohort"]
    subs, idx = np.unique(g, return_index=True)
    keep = [i for i in idx if str(co[i]) == "CAUEEG"]
    if not keep:
        return None
    labs = [str(lb[i]).upper() for i in keep]
    impaired = sum(1 for a in labs if a not in ("CN", "C", "CONTROL"))
    return impaired / len(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    tag = args.tag

    p_train = training_prevalence(tag)
    if p_train is None:
        raise SystemExit("could not compute the training prevalence; is the feature table built?")
    print("training-cohort impaired prevalence: %.4f" % p_train)

    pdir = OUT / "predictions" / tag
    by_site = {}
    for f in sorted(pdir.glob("*.npz")):
        name, site = f.stem.split("__", 1)
        by_site.setdefault(site, {})[name] = np.load(f, allow_pickle=False)

    report = {"tag": tag, "train_prevalence": p_train,
              "rules": {"target": "quantile at the target site's own prevalence (uses target labels)",
                        "source": "quantile at the training cohort's prevalence (label-free at target)",
                        "youden": "sensitivity + specificity maximised on the target (oracle)"},
              "sites": {}}

    for site, models in sorted(by_site.items()):
        target = SITE_TARGET.get(site)
        if target is None:
            continue
        ref = next((k for k in models if k.replace(" ", "_") == "random_forest"), None)
        if ref is None:
            continue
        y3 = models[ref]["y3"]
        mt, _ = target_subset(y3, target)
        y = (y3[mt] != CN).astype(int)
        if len(np.unique(y)) < 2:
            continue
        label = "%s|%s" % ("OpenNeuro" if site.startswith("ds004504") else site, target)
        p_target = float(y.mean())
        srep = {"n": int(mt.sum()), "target_prevalence": p_target, "models": {}}
        print("\n%s  n=%d  target prevalence %.3f  (training %.3f)"
              % (label, int(mt.sum()), p_target, p_train))

        for want in MODELS:
            key = next((k for k in models if k.replace(" ", "_") == want), None)
            if key is None:
                continue
            s = pairwise_score(models[key]["posterior"][mt], target)
            thr = {"target": float(np.quantile(s, 1 - p_target)),
                   "source": float(np.quantile(s, 1 - p_train)),
                   "youden": youden_threshold(y, s)}
            ops = {r: operating(y, s, t) for r, t in thr.items()}
            srep["models"][want] = {"auc": safe_auc(y, s), "operating": ops,
                                    "cost_of_unknown_prevalence": {
                                        "balanced_accuracy":
                                            ops["source"]["balanced_accuracy"]
                                            - ops["target"]["balanced_accuracy"],
                                        "sensitivity":
                                            ops["source"]["sensitivity"] - ops["target"]["sensitivity"],
                                        "specificity":
                                            ops["source"]["specificity"] - ops["target"]["specificity"]}}
            print("  %-20s AUC %.3f" % (want, safe_auc(y, s)))
            for r in ("target", "source", "youden"):
                o = ops[r]
                print("     %-7s thr %.3f  sens %.3f  spec %.3f  PPV %.3f  NPV %.3f  bal.acc %.3f"
                      % (r, o["threshold"], o["sensitivity"], o["specificity"],
                         o["ppv"], o["npv"], o["balanced_accuracy"]))
        report["sites"][label] = srep

    dest = write_result(OUT / ("threshold_sensitivity_%s.json" % tag), report,
                        script=Path(__file__).name)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
