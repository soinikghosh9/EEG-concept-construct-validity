"""Construct validity of every concept on the harmonised corpus.

The question that decides a whole Results section: once ocular, cardiac, line and channel-noise
components have been removed by ICA, do the concepts still fail their construct check?

Four cells, of which the two on the harmonised row are new:

                              analysis band 2-40 Hz / SEF95 / full-band entropy   |   2-20 Hz / SEF50 / 1-20 Hz
    old pipeline (no ICA)     3 concepts artefact-dominated, p < 0.0001            |   0
    harmonised (ICA)          ?                                                    |   ?

If the top-right and bottom-left cells both come back clean, then two independent interventions --
choosing an EMG-free analysis band, and removing artefact components from the signal -- each
suffice on their own. That is a stronger and more useful statement than either alone, because a
reader can adopt whichever is available to them.

Uses Williams's test for two dependent correlations sharing a variable (Steiger 1980), on the
subjects who have an MMSE score, so that the discriminant and criterion correlations are measured
on the same people. The alternative -- comparing a correlation computed on 1,529 pooled subjects
against one computed on 88 -- is not a comparison.

    python experiments/129_concept_validity_harmonised.py --features channel
    python experiments/129_concept_validity_harmonised.py --features channel_orig
Out: outputs/results/audit/concept_validity_harmonised_<features>.json
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

from _protocol_v2 import CN, DEM, MCI, DATASET_PATHS  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
CACHE = ROOT / "outputs" / "harmonised"
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


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    ra, rb = rank(a[ok]), rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def williams(r12, r13, r23, n):
    """Two dependent correlations sharing a variable. 1=concept, 2=artefact, 3=MMSE."""
    from scipy.stats import t as tdist
    if any(v is None for v in (r12, r13, r23)) or n < 10:
        return None, None
    det = 1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2.0 * r12 * r13 * r23
    num = (r12 - r13) * np.sqrt((n - 1) * (1.0 + r23))
    den = np.sqrt(2.0 * ((n - 1) / (n - 3)) * det + ((r12 + r13) ** 2 / 4.0) * (1.0 - r23) ** 3)
    if not np.isfinite(den) or den <= 0:
        return None, None
    tstat = float(num / den)
    return tstat, float(2.0 * (1.0 - tdist.cdf(abs(tstat), df=n - 3)))


def icc21(a, b):
    Y = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    ok = np.isfinite(Y).all(axis=1)
    Y = Y[ok]
    n, k = Y.shape
    if n < 10:
        return float("nan")
    g = Y.mean()
    ss_r = k * ((Y.mean(1) - g) ** 2).sum()
    ss_c = n * ((Y.mean(0) - g) ** 2).sum()
    ss_e = ((Y - g) ** 2).sum() - ss_r - ss_c
    msr, msc = ss_r / (n - 1), ss_c / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else 0.0


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return None
    rx = np.argsort(np.argsort(np.concatenate([x, y])))[:len(x)] + 1
    u = rx.sum() - len(x) * (len(x) + 1) / 2.0
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def read_mmse():
    out = {}
    for root in (ROOT / "datasets" / "openneuro_ds004504_raw", DATASET_PATHS.get("ds004504")):
        p = (root or Path()) / "participants.tsv"
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


def fmt(v, nd=3):
    return "  n/a " if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="channel",
                    help="subdirectory of outputs/harmonised_features")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import CONCEPT_NAMES, NORMAL_RANGES, aggregate_subject

    d = FEAT / args.features

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
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    # split halves for measurement precision, interleaved as elsewhere in the project
    Ca = np.stack([aggregate_subject(C_win[i[0::2]]) for i in idxs])
    Cb = np.stack([aggregate_subject(C_win[i[1::2]]) for i in idxs])
    print(f"features={args.features} | {len(subs)} subjects, {len(groups):,} windows")

    dur = {}
    for s in subs:
        side = CACHE / f"{s}.json"
        if side.exists():
            dur[s] = json.loads(side.read_text())["qc"].get("duration_s", np.nan)
    duration = np.array([dur.get(s, np.nan) for s in subs])

    I_ART = CONCEPT_NAMES.index("artefact_proxy")
    art = C[:, I_ART]
    mmse_map = read_mmse()
    m_on = np.array([mmse_map.get(str(s), np.nan) for s in subs], dtype=float)
    on = np.isfinite(m_on)
    print(f"MMSE available for {int(on.sum())} subjects\n")

    rep = {"features": args.features, "n_subjects": int(len(subs)),
           "n_with_mmse": int(on.sum()), "concepts": {}}

    print("=== normal-range concordance (cognitively normal subjects) ===")
    cn = y3 == CN
    for name, (lo, hi) in NORMAL_RANGES.items():
        j = CONCEPT_NAMES.index(name)
        v = C[cn, j]; v = v[np.isfinite(v)]
        inside = float(((v >= lo) & (v <= hi)).mean()) if len(v) else float("nan")
        rep["concepts"].setdefault(name, {})["normal_range"] = {
            "lo": lo, "hi": hi, "cn_median": float(np.median(v)), "fraction_inside": inside}
        print(f"  {name:22s} [{lo},{hi}]  median {np.median(v):7.3f}  inside {100*inside:5.1f}%")

    print(f"\n=== construct validity, n={int(on.sum())} with MMSE, Williams's test ===")
    hdr = (f"{'concept':22s} {'ICC':>6s} {'delta':>7s} {'rho_art':>8s} {'rho_MMSE':>9s} "
           f"{'diff':>7s} {'p':>7s}  verdict")
    print(hdr); print("-" * len(hdr))
    n_dom = n_unres = n_crit = 0
    for j, name in enumerate(CONCEPT_NAMES):
        e = rep["concepts"].setdefault(name, {})
        e["split_half_icc"] = icc21(Ca[:, j], Cb[:, j])
        e["cliffs_delta_dem_vs_cn"] = cliffs_delta(C[y3 == DEM, j], C[cn, j])
        e["expected_direction"] = CLINICAL_DIRECTION[name]
        e["rho_duration"] = spearman(C[:, j], duration)
        if name == "artefact_proxy":
            print(f"  {name:22s} {e['split_half_icc']:6.3f} "
                  f"{fmt(e['cliffs_delta_dem_vs_cn']):>7s}       (confound instrument)")
            continue

        vo, ao, mo = C[on, j].copy(), art[on], m_on[on].copy()
        r_art = spearman(vo, ao)
        if r_art is not None and r_art < 0:
            vo = -vo
        r12 = spearman(vo, ao); r13 = spearman(vo, mo)
        if r13 is not None and r13 < 0:
            mo = -mo; r13 = spearman(vo, mo)
        r23 = spearman(ao, mo)
        t, p = williams(r12, r13, r23, int(np.isfinite(vo).sum()))
        diff = None if (r12 is None or r13 is None) else r12 - r13
        verdict = ("unresolvable" if p is None or p >= 0.05
                   else "ARTEFACT-DOMINATED" if diff and diff > 0 else "criterion-dominated")
        n_dom += verdict == "ARTEFACT-DOMINATED"
        n_crit += verdict == "criterion-dominated"
        n_unres += verdict == "unresolvable"
        e.update({"rho_artefact_mmse_sample": spearman(C[on, j], art[on]),
                  "rho_mmse": spearman(C[on, j], m_on[on]),
                  "williams_t": t, "williams_p": p, "difference": diff, "verdict": verdict})
        print(f"  {name:22s} {e['split_half_icc']:6.3f} {fmt(e['cliffs_delta_dem_vs_cn']):>7s} "
              f"{fmt(e['rho_artefact_mmse_sample']):>8s} {fmt(e['rho_mmse']):>9s} "
              f"{fmt(diff):>7s} {(f'{p:.4f}' if p is not None else '  n/a'):>7s}  {verdict}")

    rep["summary"] = {"artefact_dominated": n_dom, "criterion_dominated": n_crit,
                      "unresolvable": n_unres}
    print(f"\n{n_dom} artefact-dominated, {n_crit} criterion-dominated, {n_unres} unresolvable")

    dest = write_result(OUT / f"concept_validity_harmonised_{args.features}.json", rep,
                        script="129_concept_validity_harmonised.py")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
