"""Paired comparisons across the whole benchmark, from saved subject-level predictions.

The benchmark currently reports twelve AUCs and their intervals. That is not a comparison. Every
model is scored on the SAME subjects, so the models are dependent and the correct statistic is a
paired one: an unpaired reading of two overlapping intervals is both underpowered and, when the
models agree on most subjects, badly misleading in the conservative direction.

This matters most exactly where the benchmark is most interesting. Across twelve models the
site-to-site rank correlation is +0.014 and the AUCs span 0.795-0.886 on 88 subjects, which is the
regime in which selecting a maximum and calling it a winner is a well-known error. A paired test
against a fixed reference, plus an explicit correction for having looked at twelve models, is what
distinguishes a result from a ranking.

Reports, per site and target:
  * every model against the random forest -- paired delta, bootstrap interval, p
  * the leaders against each other
  * a Holm-corrected view of the first family, because twelve comparisons were made
  * the best-versus-rest gap set against the spread expected from resampling alone

    python experiments/134_paired_tests.py --mode channel
Out: outputs/results/audit/paired_tests_<mode>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, TARGETS, pairwise_score, safe_auc, target_subset  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
N_BOOT, SEED = 10000, 42
REFERENCE = "random_forest"
SITE_TARGET = {"ds004504_raw": "CN_vs_dementia", "GENEEG": "CN_vs_MCI"}


def fmt(v, nd=3):
    return "   n/a" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def paired(y, a, b, n_boot=N_BOOT, seed=SEED):
    """Subject-clustered paired bootstrap of AUC(a) - AUC(b) on identical subjects."""
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
    return {"auc_a": a0, "auc_b": b0, "delta": a0 - b0,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": max(p, 1.0 / max(len(d), 1))}


def holm(pairs):
    """Holm-Bonferroni. Returns {name: adjusted p}, order-preserving and monotone."""
    items = sorted(((k, v) for k, v in pairs.items() if v is not None), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
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
    pdir = OUT / "predictions" / tag
    if not pdir.exists():
        raise SystemExit(f"no predictions in {pdir}; run 128 and 131 with saving enabled first")

    files = sorted(pdir.glob("*.npz"))
    by_site: dict = {}
    for f in files:
        name, site = f.stem.split("__", 1)
        by_site.setdefault(site, {})[name] = np.load(f, allow_pickle=False)
    print(f"mode={args.mode} | {len(files)} prediction files, "
          f"{len(by_site)} sites")

    report = {"mode": args.mode, "tag": tag, "n_boot": N_BOOT, "reference": REFERENCE, "sites": {}}

    for site, models in sorted(by_site.items()):
        target = SITE_TARGET.get(site)
        if target is None:
            continue
        ref = next((k for k in models if k.replace(" ", "_") == REFERENCE), None)
        if ref is None:
            print(f"[skip] {site}: no {REFERENCE} predictions")
            continue

        # every model must cover the same subjects, in the same order, or the pairing is a lie
        subj0 = models[ref]["subjects"]
        keep = {}
        for k, z in models.items():
            if len(z["subjects"]) == len(subj0) and (z["subjects"] == subj0).all():
                keep[k] = z
            else:
                print(f"  [skip] {k}: subject set differs from the reference")
        y3 = models[ref]["y3"]
        mt, yt = target_subset(y3, target)
        if mt.sum() < 16 or len(np.unique(yt)) < 2:
            continue
        scores = {k: pairwise_score(z["posterior"][mt], target) for k, z in keep.items()}
        label = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{target}"
        print(f"\n{'='*84}\n{label}  (n={int(mt.sum())}, {len(scores)} models)")

        # ---- family 1: everything against the reference --------------------------------
        print(f"\n  vs {REFERENCE}:")
        print(f"  {'model':22s} {'AUC':>7s} {'delta':>8s} {'95% CI':>19s} {'p':>8s} {'p_holm':>8s}")
        fam, raw_p = {}, {}
        for k in sorted(scores, key=lambda x: -(safe_auc(yt, scores[x]) or 0)):
            if k == ref:
                continue
            r = paired(yt, scores[k], scores[ref])
            if r is None:
                continue
            fam[k] = r
            raw_p[k] = r["p"]
        adj = holm(raw_p)
        a_ref = safe_auc(yt, scores[ref])
        print(f"  {'random forest (ref)':22s} {a_ref:7.3f}")
        for k, r in sorted(fam.items(), key=lambda kv: -kv[1]["auc_a"]):
            star = " *" if adj[k] < 0.05 else ""
            print(f"  {k:22s} {r['auc_a']:7.3f} {fmt(r['delta']):>8s} "
                  f"[{fmt(r['ci95'][0])},{fmt(r['ci95'][1])}] {r['p']:8.4f} "
                  f"{adj[k]:8.4f}{star}")
        for k in fam:
            fam[k]["p_holm"] = adj[k]

        # ---- family 2: the leaders against each other ----------------------------------
        top = sorted(scores, key=lambda x: -(safe_auc(yt, scores[x]) or 0))[:4]
        print(f"\n  leaders, pairwise:")
        head = {}
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r = paired(yt, scores[top[i]], scores[top[j]])
                if r is None:
                    continue
                head[f"{top[i]} vs {top[j]}"] = r
                print(f"  {top[i]:20s} vs {top[j]:20s} {fmt(r['delta'])} "
                      f"[{fmt(r['ci95'][0])},{fmt(r['ci95'][1])}] p={r['p']:.3f}")

        # ---- how large a gap does resampling alone produce? -----------------------------
        # The benchmark's spread has to be judged against the spread expected when every model
        # is equally good. Bootstrapping the observed AUCs jointly gives that null directly.
        rng = np.random.default_rng(SEED)
        arr = np.column_stack([scores[k] for k in scores])
        gaps = []
        for _ in range(2000):
            i = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[i])) < 2:
                continue
            a = [safe_auc(yt[i], arr[i, c]) for c in range(arr.shape[1])]
            a = [x for x in a if x is not None]
            if a:
                gaps.append(max(a) - min(a))
        obs = max(safe_auc(yt, s) or 0 for s in scores.values()) - \
            min(safe_auc(yt, s) or 1 for s in scores.values())
        gp = float(np.mean(np.asarray(gaps) >= obs)) if gaps else None
        print(f"\n  observed best-worst spread {obs:.3f}; "
              f"median spread under resampling {np.median(gaps):.3f} "
              f"(P(spread >= observed) = {gp:.3f})" if gaps else "")

        report["sites"][label] = {"n": int(mt.sum()), "reference_auc": a_ref,
                                 "vs_reference": fam, "leaders": head,
                                 "spread_observed": float(obs),
                                 "spread_null_median": float(np.median(gaps)) if gaps else None,
                                 "spread_p": gp,
                                 # the null distribution itself, so a figure can plot the real
                                 # thing rather than a parametric stand-in for it
                                 "spread_null_draws": [float(g) for g in gaps]}

    dest = write_result(OUT / f"paired_tests_{tag}.json", report,
                        script="134_paired_tests.py")
    print(f"\nwrote {dest}")
    print("\nRead the Holm-adjusted column, not the raw one: twelve models were compared against "
          "the\nsame reference on the same 88 subjects. And read the spread line -- if the "
          "observed\nbest-to-worst gap is unremarkable under resampling, the ranking is not a "
          "finding.")


if __name__ == "__main__":
    main()
