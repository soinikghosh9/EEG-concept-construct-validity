"""Check 6 as an effect size with an interval, instead of a binary verdict.

The discriminant check asks whether a concept tracks its clinical criterion more closely than it
tracks measured artefact, by Williams's test for two dependent correlations. It is reported in this
work as a three-way verdict -- criterion-dominated, artefact-dominated, or unresolvable -- and
twelve of the sixteen quantities land in the third box. Reported that way the battery looks as
though it learned nothing about twelve concepts.

That is a reporting artefact, and this script measures how much of one. Two facts sit underneath
the twelve:

  * twelve of the sixteen have a criterion correlation that is individually significant at n=87,
    so the concepts are not uninformative about cognition; what is unresolved is the COMPARISON
    against artefact, not criterion validity itself;
  * the twelve split into concepts whose difference is large but imprecise (slowing ratio +0.291,
    anterior-posterior gradient +0.251, rhythm frequency +0.218, relative theta +0.213, aperiodic
    exponent +0.166, all in the criterion direction) and concepts whose two correlations are
    genuinely almost equal (seven, all |difference| < 0.15).

Those are different scientific situations and the binary verdict merges them. The paper already
argues this point for model comparison, where it refuses to read a non-significant difference as
evidence of similarity and reports equivalence tests instead; the same discipline belongs here.

This computes, for every quantity, the paired difference |r_criterion| - |r_artefact| with a
subject-bootstrap interval, so each concept is reported as an effect with its uncertainty. The
verdict is retained as a summary of where the interval sits, not as the primary output.

    python experiments/174_discriminant_effect_sizes.py --tag channel_clean
Out: outputs/results/audit/discriminant_effect_sizes_<tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result       # noqa: E402
from src.data.feature_table import check_table      # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
RAW_ON = ROOT / "datasets" / "openneuro_ds004504_raw"
ALT_ON = ROOT / "datasets" / "openneuro_ds004504"
SEED, N_BOOT = 42, 5000


def read_mmse():
    out = {}
    for base in (RAW_ON, ALT_ON):
        p = base / "participants.tsv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                k = next((k for k in row if k.strip().upper() == "MMSE"), None)
                pid = (row.get("participant_id") or "").strip()
                if k and pid:
                    try:
                        out[pid] = float(row[k])
                    except (TypeError, ValueError):
                        pass
        if out:
            break
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    r = lambda v: np.argsort(np.argsort(v)).astype(float)          # noqa: E731
    return float(np.corrcoef(r(a[ok]), r(b[ok]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import CONCEPT_NAMES, aggregate_subject

    d = FEAT / args.tag
    check_table(d)
    C_win = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])

    mmse_map = read_mmse()
    import re
    def pid(s):
        m = re.search(r"(sub-\d+)", str(s))
        return m.group(1) if m else str(s)

    m_on = np.array([mmse_map.get(pid(s), np.nan) for s in subs], dtype=float)
    j_art = CONCEPT_NAMES.index("artefact_proxy")
    art = C[:, j_art]
    keep = np.isfinite(m_on) & np.isfinite(art)
    n = int(keep.sum())
    print(f"corpus {args.tag}: discriminant check on n = {n} subjects\n")
    if n < 30:
        raise SystemExit("MMSE not resolvable for enough subjects")

    crit_floor = 1.959964 / np.sqrt(n - 1)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    report = {"tag": args.tag, "n": n, "n_boot": N_BOOT,
              "single_correlation_detection_floor": float(crit_floor),
              "note": "Check 6 reported as an effect size. `difference` is |rho_criterion| - "
                      "|rho_artefact|; positive favours the construct. The categories stratify "
                      "the verdicts Williams's test returns as unresolvable, which merge two "
                      "different situations: a large difference the sample cannot resolve, and "
                      "two correlations that are genuinely almost equal.",
              "bootstrap_caveat":
                  "ci95_difference is reported for completeness and is NOT the inference. "
                  "Bootstrapping |r| is upward-biased when the true correlation is near zero, "
                  "because |r| is bounded below by zero and resampling noise can only raise it; "
                  "measured bias on synthetic data at n=87 is about +0.05 at rho=0 and negligible "
                  "by rho=0.10. The interval on the DIFFERENCE therefore shrinks toward zero "
                  "whenever either correlation is small, which is exactly the case for spectral "
                  "entropy (rho_criterion = -0.002) and log_n_windows (rho_artefact = +0.067). "
                  "Williams's test remains the pre-specified primary inference for check 6.",
              "concepts": {}}

    # The pre-specified Williams verdict, so the two can be reported side by side.
    wpath = OUT / f"concept_validity_harmonised_{args.tag}.json"
    williams = {}
    if wpath.exists():
        wj = json.loads(wpath.read_text())
        for k, v in wj.get("concepts", {}).items():
            williams[k] = {"verdict": v.get("verdict"), "p": v.get("williams_p")}

    A, M = art[keep], m_on[keep]
    rows = []
    for name in CONCEPT_NAMES:
        if name in ("artefact_proxy",):
            continue
        j = CONCEPT_NAMES.index(name)
        V = C[keep, j]
        if not np.isfinite(V).sum() >= 10:
            continue
        r_c, r_a = spearman(V, M), spearman(V, A)
        if r_c is None or r_a is None:
            continue
        diff = abs(r_c) - abs(r_a)
        b = []
        for bi in boot_idx:
            rc, ra = spearman(V[bi], M[bi]), spearman(V[bi], A[bi])
            if rc is not None and ra is not None:
                b.append(abs(rc) - abs(ra))
        lo, hi = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) if b else (None, None)
        # classification that separates "no evidence" from "imprecise but directional"
        # Categories use the point estimate and the pre-specified Williams verdict, not the
        # bootstrap interval, for the bias reason recorded in `bootstrap_caveat`.
        if abs(diff) >= 0.15:
            cat = "directional, underpowered"
        elif max(abs(r_c), abs(r_a)) < crit_floor:
            cat = "uninformative on both axes"
        else:
            cat = "correlations comparable"
        report["concepts"][name] = {
            "rho_criterion": r_c, "rho_artefact": r_a, "difference": diff,
            "ci95_difference": [lo, hi],
            "criterion_individually_significant": bool(abs(r_c) >= crit_floor),
            "effect_category": cat,
            "williams_verdict": williams.get(name, {}).get("verdict"),
            "williams_p": williams.get(name, {}).get("p")}
        rows.append((name, r_c, r_a, diff, lo, hi, cat,
                     williams.get(name, {}).get("verdict")))

    rows.sort(key=lambda r: -r[3])
    print(f"{'concept':22s} {'r_crit':>7s} {'r_art':>7s} {'diff':>7s}  "
          f"{'Williams':>20s}  effect category")
    for name, rc, ra, diff, lo, hi, cat, wv in rows:
        print(f"{name:22s} {rc:+7.3f} {ra:+7.3f} {diff:+7.3f}  {str(wv):>20s}  {cat}")

    # what the Williams-unresolvable verdicts actually contain
    unres = [r for r in rows if r[7] == "unresolvable"]
    sub = {}
    for r in unres:
        sub[r[6]] = sub.get(r[6], 0) + 1
    report["williams_unresolvable_breakdown"] = sub
    print(f"\nWilliams returns 'unresolvable' for {len(unres)} quantities. They contain:")
    for k, v in sorted(sub.items(), key=lambda x: -x[1]):
        print(f"   {k:30s} {v}")

    tally = {}
    for r in rows:
        tally[r[6]] = tally.get(r[6], 0) + 1
    report["tally"] = tally
    n_sig_crit = sum(1 for r in rows if abs(r[1]) >= crit_floor)
    report["n_criterion_individually_significant"] = n_sig_crit
    print("\ncategory tally:")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"   {k:30s} {v}")
    print(f"\nconcepts whose criterion correlation is individually significant "
          f"(|rho| >= {crit_floor:.3f}): {n_sig_crit} of {len(rows)}")

    dest = write_result(OUT / f"discriminant_effect_sizes_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
