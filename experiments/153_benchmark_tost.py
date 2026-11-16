"""Equivalence testing for the benchmark, by two one-sided tests.

The benchmark reports that no paired comparison against the reference reaches significance, and
that the observed best-to-worst spread is smaller than resampling alone produces. Neither statement
is an equivalence claim. A non-significant difference is absence of evidence; the manuscript's
sentence -- that the models are not separable -- asserts evidence of absence, and that requires a
margin and a test against it.

This runs TOST on the same saved subject-level predictions, with the same subject-clustered paired
bootstrap, the same seed and the same reference as 134_paired_tests.py, so the deltas here are
identical to the ones already reported and only the inference added is new.

For a margin D, equivalence at alpha is concluded when the (1 - 2*alpha) interval of the paired
delta lies entirely inside (-D, +D). The reported p is the larger of the two one-sided bootstrap
tail probabilities, so p < alpha and interval inclusion agree.

Margins follow the convention already used in outputs/results/ceiling_tost.json: 0.05, 0.075, 0.10.
0.05 in AUC is the smaller of the two differences a screening instrument could plausibly be chosen
on; 0.10 is frankly permissive and is reported so the reader can see where the conclusion turns.

    python experiments/145_benchmark_tost.py --mode channel --tag channel_clean
Out: outputs/results/audit/benchmark_tost_<tag>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import TARGETS, pairwise_score, safe_auc, target_subset  # noqa: E402,F401
from src.utils.provenance import write_result  # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
N_BOOT, SEED = 10000, 42
REFERENCE = "random_forest"
MARGINS = (0.05, 0.075, 0.10)
ALPHA = 0.05
SITE_TARGET = {"ds004504_raw": "CN_vs_dementia", "GENEEG": "CN_vs_MCI"}


def paired_delta_draws(y, a, b, n_boot=N_BOOT, seed=SEED):
    """Subject-clustered paired bootstrap draws of AUC(a) - AUC(b). Identical to 134's `paired`."""
    y = np.asarray(y)
    a0, b0 = safe_auc(y, a), safe_auc(y, b)
    if a0 is None or b0 is None:
        return None, None, None
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
    return a0 - b0, np.asarray(out), (a0, b0)


def tost(draws, margin, alpha=ALPHA):
    """Two one-sided bootstrap tests. Equivalent when the (1-2a) interval sits inside +/- margin."""
    lo = float(np.percentile(draws, 100 * alpha))
    hi = float(np.percentile(draws, 100 * (1 - alpha)))
    p_lower = float((draws <= -margin).mean())   # against H0: delta <= -margin
    p_upper = float((draws >= margin).mean())    # against H0: delta >= +margin
    p = max(p_lower, p_upper)
    return {"margin": margin,
            "ci90": [lo, hi],
            "p_tost": p,
            "equivalent": bool(lo > -margin and hi < margin)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.mode
    pdir = OUT / "predictions" / tag
    if not pdir.exists():
        raise SystemExit(f"no predictions in {pdir}")

    by_site: dict = {}
    for f in sorted(pdir.glob("*.npz")):
        name, site = f.stem.split("__", 1)
        by_site.setdefault(site, {})[name] = np.load(f, allow_pickle=False)

    report = {"mode": args.mode, "tag": tag, "n_boot": N_BOOT, "alpha": ALPHA,
              "margins": list(MARGINS), "reference": REFERENCE,
              "definition": "equivalence at alpha when the (1-2*alpha) paired-delta interval "
                            "lies inside +/- margin; p is the larger one-sided tail",
              "sites": {}}

    for site, models in sorted(by_site.items()):
        target = SITE_TARGET.get(site)
        if target is None:
            continue
        ref = next((k for k in models if k.replace(" ", "_") == REFERENCE), None)
        if ref is None:
            continue

        subj0 = models[ref]["subjects"]
        keep = {k: z for k, z in models.items()
                if len(z["subjects"]) == len(subj0) and (z["subjects"] == subj0).all()}
        y3 = models[ref]["y3"]
        mt, yt = target_subset(y3, target)
        if mt.sum() < 16 or len(np.unique(yt)) < 2:
            continue
        scores = {k: pairwise_score(z["posterior"][mt], target) for k, z in keep.items()}
        label = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{target}"

        srep = {"n": int(mt.sum()), "reference_auc": safe_auc(yt, scores[ref]), "models": {}}
        print(f"\n{'='*92}\n{label}  (n={int(mt.sum())})")
        head = "  ".join(f"D={m:g}".rjust(11) for m in MARGINS)
        print(f"  {'model':22s} {'AUC':>7s} {'delta':>8s}   {head}")

        for k in sorted(scores, key=lambda x: -(safe_auc(yt, scores[x]) or 0)):
            if k == ref:
                continue
            delta, draws, aucs = paired_delta_draws(yt, scores[k], scores[ref])
            if draws is None or len(draws) == 0:
                continue
            per = {f"{m:g}": tost(draws, m) for m in MARGINS}
            srep["models"][k] = {"auc": aucs[0], "delta": delta,
                                 "ci90": per[f"{MARGINS[0]:g}"]["ci90"], "tost": per}
            cells = "  ".join(
                ("EQUIV" if per[f'{m:g}']['equivalent'] else "  --").rjust(6)
                + f"{per[f'{m:g}']['p_tost']:.3f}".rjust(5) for m in MARGINS)
            print(f"  {k:22s} {aucs[0]:7.3f} {delta:+8.3f}   {cells}")

        # the sentence the manuscript actually makes: every model against the reference
        for m in MARGINS:
            key = f"{m:g}"
            all_eq = all(v["tost"][key]["equivalent"] for v in srep["models"].values())
            worst = max((v["tost"][key]["p_tost"] for v in srep["models"].values()), default=None)
            srep.setdefault("family", {})[key] = {
                "all_models_equivalent": bool(all_eq),
                "max_p_tost": worst,
                "n_models": len(srep["models"])}
            print(f"  [family] margin {m:g}: all equivalent = {all_eq}   max p = {worst:.4f}")
        report["sites"][label] = srep

    dest = write_result(OUT / f"benchmark_tost_{tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
