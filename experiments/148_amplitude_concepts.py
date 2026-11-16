"""Repair attempt for the inexpressible-construct failure, and the battery applied to it.

Three concepts are named after inter-electrode amplitude statements -- alpha is maximal
occipitally, the anterior-posterior gradient is lost, interhemispheric amplitude asymmetry exceeds
35 to 50 per cent -- and all three are computed from relative band power, a within-channel ratio.
They are exactly invariant to per-channel rescaling and therefore cannot measure what they are
named after. This is the only failure mode in the battery for which a repair was available but
untried.

The repair is to compute the same three constructs from ABSOLUTE band power
(`ConceptExtractor.amplitude_concepts`). Each remains a ratio between electrodes, so a
recording-level gain still cancels, but the inter-electrode contrast survives.

It only works if the pipeline preserved inter-channel amplitude in the first place, which is what
the scaling mode decides. This script therefore runs the whole thing twice:

  * **global scaling** -- one divisor for the recording, inter-channel amplitude intact. Treatment.
  * **per-channel scaling** -- each channel divided by its own IQR. Negative control, where the
    quantity is destroyed before it is computed and the repair must fail.

A repair that looked equally good in both arms would be measuring something other than amplitude.

Four questions, in order:
  1. Do the new estimators respond to per-channel gain where the old ones do not? (decisive)
  2. Do they carry criterion validity against MMSE, and do they beat the relative versions?
  3. Are they precise enough to use (split-half), and do they separate known groups?
  4. Do they add anything to the classifier?

    python experiments/148_amplitude_concepts.py
Out: outputs/results/audit/amplitude_concepts.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, safe_auc, target_subset, pairwise_score  # noqa: E402
from src.utils.provenance import write_result                                   # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

CACHE = ROOT / "outputs" / "harmonised"
FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
RAW_ON = ROOT / "datasets" / "openneuro_ds004504_raw"

WINDOW, STRIDE, MAX_WINDOWS = 1024, 1024, 50
REJECT_V, SFREQ = 200e-6, 256.0
#: The five cohorts of the corpus every other result in the paper is computed on.
MAIN_COHORTS = {"CAUEEG", "GENEEG", "MISP", "P-ADIC", "ds004504_raw"}

#: The acquisition-matching protocol of the manuscript, replicated from the invocation that built
#: `harmonised_features/channel_clean` (run_extract_clean.cmd). Under `--matched` this script
#: reproduces that corpus exactly -- same truncation, same K, same per-subject seeded choice -- so
#: the amplitude estimators are measured on the identical subjects and identical windows as the
#: relative estimators they are compared against. Without it the arms differ by 31 subjects and
#: roughly 30,000 windows, and the comparison confounds the estimator with the corpus.
TRUNC_MATCHED = {"CAUEEG": 235.0, "ds004504_raw": 235.0, "GENEEG": 235.0,
                 "MISP": 235.0, "P-ADIC": 235.0}
FIXED_WINDOWS = 24
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "Healthy": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}


def window_starts(n_samples: int, sid: str):
    if n_samples < WINDOW:
        return []
    starts = list(range(0, n_samples - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        seed = int(hashlib.md5(str(sid).encode("utf-8")).hexdigest()[:8], 16)
        starts = sorted(np.random.default_rng(seed).choice(starts, MAX_WINDOWS, replace=False))
    return list(starts)


def read_mmse():
    out = {}
    p = RAW_ON / "participants.tsv"
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                out[row["participant_id"].strip()] = float(row["MMSE"])
            except (TypeError, ValueError, KeyError):
                pass
    return out


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(a[m], b[m]).statistic
    return None if not np.isfinite(r) else float(r)


def williams(r12, r13, r23, n):
    """Williams's test for two dependent correlations sharing a variable (Steiger 1980)."""
    if any(v is None for v in (r12, r13, r23)) or n < 10:
        return None, None
    from scipy.stats import t as tdist
    d = r12 - r13
    det = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    if det <= 0:
        return None, None
    rbar = (r12 + r13) / 2
    denom = np.sqrt(2 * (n - 1) / (n - 3) * det + rbar**2 * (1 - r23)**3)
    if denom <= 0:
        return None, None
    tv = d * np.sqrt((n - 1) * (1 + r23) / denom)
    return float(tv), float(2 * (1 - tdist.cdf(abs(tv), n - 3)))


def cliffs_delta(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


# ============================================================== 1. the perturbation check
def perturbation_check(ex, seed=0):
    """Do the new estimators respond to per-channel gain where the relative ones do not?

    A synthetic window with a deliberate posterior-to-anterior amplitude gradient, then random
    per-channel gains applied. The construct genuinely changes; a valid estimator of it must move.
    """
    rng = np.random.default_rng(seed)
    n_ch, n_t = 19, WINDOW
    t = np.arange(n_t) / SFREQ
    # Built at unit scale, not in volts. The extractor always sees robust-scaled input, and the
    # epsilon guarding the PSD normalisation is negligible there. Feeding raw microvolt-scale data
    # makes that epsilon dominate the per-channel total and breaks the very gain invariance this
    # check is testing -- an artefact of the test, not a property of the estimator.
    x = rng.normal(0, 1.0, (n_ch, n_t))
    # posterior channels (P3, P4, O1, O2, T5, T6 in the default montage order) get large alpha
    big = [6, 7, 8, 9, 14, 15]
    for c in range(n_ch):
        amp = 40.0 if c in big else 6.0
        x[c] += amp * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))

    X0 = torch.from_numpy(x[None].astype(np.float32))
    gains = rng.uniform(0.25, 4.0, n_ch).astype(np.float32)
    X1 = torch.from_numpy((x * gains[:, None])[None].astype(np.float32))

    rel0 = ex.window_concepts(X0).numpy()[0]
    rel1 = ex.window_concepts(X1).numpy()[0]
    amp0 = ex.amplitude_concepts(X0).numpy()[0]
    amp1 = ex.amplitude_concepts(X1).numpy()[0]

    from src.models.v6 import AMPLITUDE_CONCEPT_NAMES, CONCEPT_NAMES
    idx = {n: i for i, n in enumerate(CONCEPT_NAMES)}
    relevant = ["posterior_maximality", "ap_gradient_alpha", "asymmetry_alpha"]
    return {
        "gains_applied": [round(float(g), 3) for g in gains],
        "relative_versions": {n: {"before": float(rel0[idx[n]]), "after": float(rel1[idx[n]]),
                                  "abs_change": float(abs(rel1[idx[n]] - rel0[idx[n]]))}
                              for n in relevant},
        "amplitude_versions": {n: {"before": float(amp0[i]), "after": float(amp1[i]),
                                   "abs_change": float(abs(amp1[i] - amp0[i]))}
                               for i, n in enumerate(AMPLITUDE_CONCEPT_NAMES)},
    }


# ============================================================== 2. corpus extraction
def extract(mode, limit=None, matched=False):
    from src.data.scaling import robust_scale
    from src.models.v6 import ConceptExtractor

    ex = ConceptExtractor()
    # The harmonised cache has since grown to seven cohorts. The relative concepts these are
    # compared against were computed on the five-cohort corpus, so restrict to it: otherwise the
    # amplitude and relative versions would be measured on different subject sets and every
    # comparison below would confound the estimator with the corpus.
    sides = [s for s in sorted(CACHE.glob("*.json"))
             if json.loads(s.read_text()).get("cohort") in MAIN_COHORTS]
    if limit:
        sides = sides[:limit]
    rows, groups, labels, cohorts = [], [], [], []
    quota_dropped = []
    t0 = time.time()
    for k, side in enumerate(sides, 1):
        meta = json.loads(side.read_text())
        sid, cohort, grp = meta["subject"], meta["cohort"], meta.get("group", "?")
        npy = side.with_suffix(".npy")
        if not npy.exists():
            continue
        data = np.load(npy)
        # Truncate BEFORE windowing. Selecting windows first and trimming afterwards leaves long
        # recordings with a handful and short ones with all of theirs, which is the confound the
        # protocol exists to remove.
        if matched and cohort in TRUNC_MATCHED:
            data = data[:, :int(TRUNC_MATCHED[cohort] * SFREQ)]
        keep = [w for s in window_starts(data.shape[1], sid)
                if float(np.ptp(w := data[:, s:s + WINDOW], axis=1).max()) <= REJECT_V]
        if matched:
            if len(keep) < FIXED_WINDOWS:
                quota_dropped.append((sid, cohort, grp, len(keep)))
                continue
            # Byte-identical to 127_extract_harmonised.py's quota, so the subject set and the
            # window set both reproduce `channel_clean` exactly rather than approximately.
            r = np.random.default_rng(
                int(hashlib.md5(("q" + str(sid)).encode("utf-8")).hexdigest()[:8], 16))
            pick = sorted(r.choice(len(keep), FIXED_WINDOWS, replace=False))
            keep = [keep[i] for i in pick]
        if not keep:
            continue
        W = np.stack([robust_scale(w, mode) for w in keep]).astype(np.float32)
        with torch.no_grad():
            rows.append(ex.amplitude_concepts(torch.from_numpy(W)).cpu().numpy())
        groups += [sid] * len(keep)
        labels += [grp] * len(keep)
        cohorts += [cohort] * len(keep)
        if k % 250 == 0 or k == len(sides):
            el = (time.time() - t0) / 60
            print(f"  [{mode}] {k}/{len(sides)}  {len(groups):,} windows  {el:.1f} min",
                  flush=True)
    if matched:
        print(f"  [{mode}] K={FIXED_WINDOWS} quota dropped {len(quota_dropped)} subjects")
    return (np.concatenate(rows).astype(np.float32), np.array(groups), np.array(labels),
            np.array(cohorts))


def per_subject(A, groups):
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    with np.errstate(invalid="ignore"):
        C = np.stack([np.nanmedian(A[i], axis=0) for i in idxs])
    halves = []
    for i in idxs:
        h = len(i) // 2
        with np.errstate(invalid="ignore"):
            halves.append((np.nanmedian(A[i[:h]], axis=0) if h else np.full(A.shape[1], np.nan),
                           np.nanmedian(A[i[h:]], axis=0) if h else np.full(A.shape[1], np.nan)))
    H1 = np.stack([a for a, _ in halves])
    H2 = np.stack([b for _, b in halves])
    return subs, idxs, C, H1, H2


def icc21(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return None
    x, y = a[m], b[m]
    mu = np.concatenate([x, y]).mean()
    bms = 2 * ((x + y) / 2 - mu).var(ddof=1)
    wms = ((x - y) ** 2 / 2).mean()
    den = bms + wms
    return None if den <= 0 else float((bms - wms) / den)


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--matched", action="store_true",
                    help="apply the manuscript's acquisition-matching protocol (truncate to a "
                         "common 235 s, then exactly K=24 clean windows per subject) so the "
                         "amplitude estimators are measured on the same 1,296 subjects and "
                         "31,104 windows as every other reported result. Without it the arm "
                         "runs on the pre-quota 1,327 subjects and 61,582 windows, which is "
                         "exposed to the epoch-count confound the protocol removes.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import AMPLITUDE_CONCEPT_NAMES, CONCEPT_NAMES, ConceptExtractor

    # The matched run is written to its own file and joined against the matched relative arm.
    # Overwriting the unmatched result would destroy the comparison that shows what the
    # matching is worth.
    suffix = "_clean" if args.matched else ""
    ref_feat = "channel_clean" if args.matched else "channel"
    ref_validity = (f"concept_validity_harmonised_channel{suffix}.json")

    report = {"note": "Repair attempt for the inexpressible-construct failure mode.",
              "corpus": ("matched: 235 s truncation + K=24 per subject, identical to the "
                         "channel_clean corpus every other result uses"
                         if args.matched else
                         "unmatched: up to 50 windows per subject, no duration truncation"),
              "matched": bool(args.matched),
              "artefact_proxy_from": ref_feat,
              "relative_reference": ref_validity,
              "concepts": AMPLITUDE_CONCEPT_NAMES}

    # ---- 1. perturbation -------------------------------------------------------------
    print("=" * 84)
    print("1. Perturbation check: does the estimator move when the construct changes?")
    print("=" * 84)
    ex = ConceptExtractor()
    pert = perturbation_check(ex)
    report["perturbation"] = pert
    for n, v in pert["relative_versions"].items():
        print(f"  relative  {n:24s} {v['before']:+.5f} -> {v['after']:+.5f}   "
              f"|change| {v['abs_change']:.2e}")
    for n, v in pert["amplitude_versions"].items():
        print(f"  amplitude {n:24s} {v['before']:+.5f} -> {v['after']:+.5f}   "
              f"|change| {v['abs_change']:.2e}")
    rel_max = max(v["abs_change"] for v in pert["relative_versions"].values())
    amp_min = min(v["abs_change"] for v in pert["amplitude_versions"].values())
    report["perturbation"]["verdict"] = (
        "PASS: amplitude versions respond, relative versions do not"
        if amp_min > 100 * max(rel_max, 1e-12) else "INCONCLUSIVE")
    print(f"\n  relative max |change| {rel_max:.2e}   amplitude min |change| {amp_min:.2e}")
    print(f"  verdict: {report['perturbation']['verdict']}")

    # ---- 2. corpus, both scaling arms -------------------------------------------------
    mmse_map = read_mmse()
    Cch = json.loads((OUT / ref_validity).read_text())
    report["arms"] = {}

    ref = None
    for mode in ("global", "channel"):
        print("\n" + "=" * 84)
        print(f"2. Corpus extraction under {mode} scaling"
              f"{'  (negative control)' if mode == 'channel' else ''}"
              f"{'  [matched corpus]' if args.matched else ''}")
        print("=" * 84)
        A, groups, labels, cohorts = extract(mode, args.limit, matched=args.matched)
        subs, idxs, C, H1, H2 = per_subject(A, groups)
        sub_lab = np.array([labels[i[0]] for i in idxs])
        sub_coh = np.array([cohorts[i[0]] for i in idxs])
        y3 = np.array([LABEL3.get(l, -1) for l in sub_lab])

        # artefact proxy on the same subjects, from the existing channel-mode features
        # The window array has 15 columns in its own order; CONCEPT_NAMES has 17 and includes two
        # subject-level quantities. Go through aggregate_subject rather than indexing the window
        # array with a CONCEPT_NAMES position, which silently reads the wrong column.
        from src.models.v6 import aggregate_subject
        check_table(FEAT / ref_feat)   # artefact proxy is joined to these windows
        fm = np.load(FEAT / ref_feat / "meta.npz", allow_pickle=False)
        fc = np.load(FEAT / ref_feat / "window_concepts.npy")
        fg = fm["groups"]
        j = CONCEPT_NAMES.index("artefact_proxy")
        art_by_sub = {str(s): float(aggregate_subject(fc[fg == s])[j]) for s in np.unique(fg)}

        m_on = np.array([mmse_map.get(str(s), np.nan) for s in subs], dtype=float)
        art = np.array([art_by_sub.get(str(s), np.nan) for s in subs], dtype=float)
        on = np.isfinite(m_on) & np.isfinite(art)
        print(f"  {len(subs)} subjects, {len(groups):,} windows, {int(on.sum())} with MMSE")

        # A claim that this arm is "the same corpus" is worth nothing unless it is checked. The
        # subject set must be identical to the reference feature table, not merely the same size:
        # an equal count reached through a different quota outcome would still break the pairing.
        reproduces = None
        if args.matched:
            want = set(np.unique(fg).tolist())
            got = set(subs.tolist())
            reproduces = (want == got) and len(groups) == len(fg)
            if not reproduces:
                print(f"  [WARN] corpus does not reproduce {ref_feat}: "
                      f"{len(got - want)} extra, {len(want - got)} missing subjects, "
                      f"{len(groups)} vs {len(fg)} windows")
            else:
                print(f"  corpus reproduces {ref_feat} exactly "
                      f"({len(got)} subjects, {len(groups):,} windows)")

        arm = {"n_subjects": int(len(subs)), "n_windows": int(len(groups)),
               "n_with_mmse": int(on.sum()),
               "reproduces_reference_corpus": reproduces, "concepts": {}}
        for i, name in enumerate(AMPLITUDE_CONCEPT_NAMES):
            v = C[:, i]
            r_mmse = spearman(v[on], m_on[on])
            r_art = spearman(v[on], art[on])
            r_am = spearman(art[on], m_on[on])
            tv, pv = williams(abs(r_art) if r_art else None,
                              abs(r_mmse) if r_mmse else None, r_am, int(on.sum()))
            dem = v[(y3 == DEM) & np.isfinite(v)]
            cn = v[(y3 == CN) & np.isfinite(v)]
            arm["concepts"][name] = {
                "split_half_icc": icc21(H1[:, i], H2[:, i]),
                "rho_mmse": r_mmse, "rho_artefact": r_art,
                "williams_t": tv, "williams_p": pv,
                "cliffs_delta_dem_vs_cn": cliffs_delta(dem, cn),
                "median": float(np.nanmedian(v)),
            }
            e = arm["concepts"][name]

            def f(x, spec="{:+.3f}"):
                return spec.format(x) if isinstance(x, (int, float)) else "  n/a"

            print(f"  {name:26s} ICC {f(e['split_half_icc'], '{:.3f}')}  "
                  f"rho_MMSE {f(r_mmse)}  rho_art {f(r_art)}  "
                  f"delta {f(e['cliffs_delta_dem_vs_cn'])}")
        report["arms"][mode] = arm
        if mode == "global":
            ref = (subs, C, y3, sub_coh, on, m_on)

    # ---- 3. comparison against the relative versions -----------------------------------
    print("\n" + "=" * 84)
    print("3. Amplitude versions against the relative ones they replace")
    print("=" * 84)
    pairs = [("amp_posterior_maximality", "posterior_maximality"),
             ("amp_ap_gradient_alpha", "ap_gradient_alpha"),
             ("amp_asymmetry_alpha", "asymmetry_alpha")]
    comp = {}
    for a_name, r_name in pairs:
        a = report["arms"]["global"]["concepts"][a_name]
        r = Cch["concepts"].get(r_name, {})
        comp[a_name] = {"amplitude_rho_mmse": a["rho_mmse"],
                        "relative_rho_mmse": r.get("rho_mmse"),
                        "amplitude_icc": a["split_half_icc"],
                        "relative_icc": r.get("split_half_icc")}
        def g(x, spec="{:+.3f}"):
            return spec.format(x) if isinstance(x, (int, float)) else "  n/a"

        print(f"  {a_name:26s} rho_MMSE {g(a['rho_mmse'])}  vs relative "
              f"{g(r.get('rho_mmse'))}   "
              f"ICC {g(a['split_half_icc'], '{:.3f}')} vs {g(r.get('split_half_icc'), '{:.3f}')}")
    report["against_relative"] = comp

    # ---- 4. does the negative control behave? ------------------------------------------
    g = report["arms"]["global"]["concepts"]
    c = report["arms"]["channel"]["concepts"]
    control = {n: {"global_icc": g[n]["split_half_icc"], "channel_icc": c[n]["split_half_icc"],
                   "global_rho_mmse": g[n]["rho_mmse"], "channel_rho_mmse": c[n]["rho_mmse"]}
               for n in AMPLITUDE_CONCEPT_NAMES}
    report["negative_control"] = control
    print("\n" + "=" * 84)
    print("4. Negative control: per-channel scaling should destroy the quantity")
    print("=" * 84)
    for n, v in control.items():
        def h(x):
            return f"{x:+.3f}" if isinstance(x, (int, float)) else "  n/a"

        print(f"  {n:26s} rho_MMSE global {h(v['global_rho_mmse'])}  "
              f"channel {h(v['channel_rho_mmse'])}")

    dest = OUT / f"amplitude_concepts{suffix}.json"
    write_result(dest, report, script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
