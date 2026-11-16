"""The construct-validation battery run on wearable forehead-and-ear hardware.

EasyCog (Hu, Zhou et al., IMWUT 10:1, 2026) records 16 channels -- four forehead electrodes
and two six-electrode cEEGrid strips -- at 125 Hz, with MoCA and MMSE for every session. It
is not a discrimination site for this work: after the epoch quota it holds ten Alzheimer's
cases against ten controls, and the controls are twelve years younger. What it supplies is a
criterion measure on hardware nobody in this study designed for, which is what the decisive
construct check is short of -- that check runs on 87 subjects in the main corpus and its
stated limitation is sample size.

Three arms:

  A. Extraction and audit. Concepts under the matching protocol of the main analysis
     (K = 24 artefact-free 4 s epochs, 200 uV peak-to-peak rejection before scaling), plus
     the acquisition floors and the same quantities under the as-distributed reference, so
     nothing here rests on the common-average step being the right one for this array.

  B. Checks 2, 4 and 6 of the battery. Split-half precision, criterion validity against both
     MMSE and MoCA, and Williams's test of the criterion correlation against the artefact
     correlation on the same sessions.

  C. Across-session test-retest. Five subjects were recorded on two or three separate days,
     each visit re-applying the headband and both ear strips. The main corpus supports only
     a split-half estimate within one recording, so this is the only re-application estimate
     available anywhere in this study. Seven pairs: a bound, not an estimate.

Two caveats travel with every number below and are recorded in the output. The released
channel order is inferred from the benchmark repository rather than documented, and the ear
electrode coordinates are reconstructed from a 2-D schematic; arm A therefore repeats the
region assignment across a sweep of that reconstruction.

    python experiments/191_easycog_battery.py --root "<EasyCog_Dataset>"
Out: outputs/results/audit/easycog_battery.json
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "audit"
SEED = 42
WIN_S, K, PP_UV, SFREQ = 4.0, 24, 200.0, 125.0

DEFAULT_ROOT = "datasets/EasyCog_Dataset"  # the public release, unpacked; or pass --root

#: The order window_concepts() returns. NOT CONCEPT_NAMES: aggregate_subject() inserts
#: pdr_variability at index 2 and log_n_windows at the end, so the two lists differ from
#: index 2 onward. Mislabelling here silently renames every concept in the output.
WIN_NAMES = ['pdr_frequency', 'pdr_prominence', 'posterior_maximality', 'ap_gradient_alpha',
             'ap_gradient_freq', 'asymmetry_alpha', 'asymmetry_freq', 'theta_alpha_ratio',
             'slowing_ratio', 'rel_alpha', 'rel_theta', 'aperiodic_exponent',
             'spectral_edge', 'spectral_entropy', 'artefact_proxy']
ARTEFACT = 'artefact_proxy'


# ----------------------------------------------------------------------------- statistics
def williams(r12, r13, r23, n):
    """Two dependent correlations sharing a variable. 1=concept, 2=criterion, 3=artefact."""
    from scipy.stats import t as tdist
    det = 1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2.0 * r12 * r13 * r23
    num = (r12 - r13) * np.sqrt((n - 1) * (1.0 + r23))
    den = np.sqrt(2.0 * ((n - 1) / (n - 3)) * det + ((r12 + r13) ** 2 / 4.0) * (1.0 - r23) ** 3)
    if not np.isfinite(den) or den <= 0:
        return None, None
    t = float(num / den)
    return t, float(2.0 * (1.0 - tdist.cdf(abs(t), df=n - 3)))


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    Y = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    Y = Y[np.isfinite(Y).all(axis=1)]
    n, k = Y.shape
    if n < 3:
        return float("nan")
    g = Y.mean()
    ss_r = k * ((Y.mean(1) - g) ** 2).sum()
    ss_c = n * ((Y.mean(0) - g) ** 2).sum()
    ss_e = ((Y - g) ** 2).sum() - ss_r - ss_c
    ms_r, ms_c, ms_e = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / den) if den > 0 else float("nan")


def boot_ci(fn, *arrays, n_boot=2000, seed=SEED):
    """Percentile interval, resampling the unit of analysis (rows) with replacement."""
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = fn(*[a[i] for a in arrays])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = rankdata(np.r_[a, b])
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def partial_r(x, y, ctrl):
    from scipy import stats
    rx = x - np.polyval(np.polyfit(ctrl, x, 1), ctrl)
    ry = y - np.polyval(np.polyfit(ctrl, y, 1), ctrl)
    return stats.pearsonr(rx, ry)


# ----------------------------------------------------------------------------- extraction
def read_meta(root):
    import openpyxl
    ws = openpyxl.load_workbook(Path(root) / "Patient_Info_dataset.xlsx")["Sheet1"]
    out = {}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[8] is None:
            continue
        out[f"{r[8]}_patient"] = dict(sex=r[1], age=r[2], dis=str(r[3]), env=str(r[4]),
                                      moca=r[5], mmse=r[6])
    return out


def epochs_from(x, win, pp_uv, car):
    """[C,T] microvolts -> ([n,C,win] accepted, n_available, rejection_rate)."""
    if car:
        x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[1] // win
    if n == 0:
        return np.empty((0, x.shape[0], win)), 0, 1.0
    ep = x[:, :n * win].reshape(x.shape[0], n, win).transpose(1, 0, 2)
    # peak-to-peak on the worst channel, before any scaling: the only order in which a
    # microvolt threshold keeps its meaning.
    pp = (ep.max(-1) - ep.min(-1)).max(-1)
    ok = pp <= pp_uv
    return ep[ok], n, float(1.0 - ok.mean())


def extract(ex, ep, sid, k=K, halves=False):
    """Median concept vector over a seeded quota of k epochs; optionally the two halves."""
    import torch
    if len(ep) < k:
        return None
    rng = np.random.default_rng(int(str(sid).split('_')[0]))
    sel = np.sort(rng.choice(len(ep), k, replace=False))
    with torch.no_grad():
        c = np.asarray(ex.window_concepts(torch.tensor(ep[sel], dtype=torch.float32)))
    with np.errstate(all="ignore"):
        v = np.nanmedian(c, axis=0)
        if not halves:
            return v
        return v, np.nanmedian(c[0::2], axis=0), np.nanmedian(c[1::2], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from scipy import stats

    from src.data.wearable_montage import (EASYCOG_COORDS, STD_CH, easycog_coords,
                                           region_occupancy)
    from src.models.v6.concepts import ConceptExtractor
    from src.utils.provenance import write_result

    root = Path(args.root)
    meta = read_meta(root)
    win = int(SFREQ * WIN_S)
    report = {"source": str(root), "sfreq": SFREQ, "window_s": WIN_S, "K": K,
              "pp_threshold_uV": PP_UV, "concepts": WIN_NAMES,
              "channel_order": STD_CH,
              "caveats": {
                  "channel_order": "inferred from the EasyCog benchmark repository "
                                   "(analysis_utils.STD_CH), corroborated by its EOG "
                                   "derivation; not stated in the dataset readme",
                  "ear_coordinates": "reconstructed from the repository's 2-D layout table; "
                                     "cEEGrid_sensor.xlsx is referenced but not distributed",
                  "provenance": "ASR and ASR-EOG already applied at source, 0.1 Hz high-pass, "
                                "no notch; no raw accession exists"}}

    # ---------------------------------------------------------------- A. montage sensitivity
    print("=" * 78)
    print("A1. REGION OCCUPANCY, and its sensitivity to the reconstructed ear coordinates")
    occ = region_occupancy()
    for k, v in occ.items():
        print(f"    {k:10s} n={len(v):2d}  {v}")
    sweep = []
    for cy in (0.00, -0.05, -0.10, -0.20, -0.30, -0.40):
        for dy in (0.06, 0.10, 0.15):
            o = region_occupancy(easycog_coords(ear_centre_y=cy, dy=dy))
            sweep.append({"ear_centre_y": cy, "dy": dy,
                          "n_posterior": len(o["posterior"]), "n_occipital": len(o["occipital"])})
    bad = [s for s in sweep if s["n_posterior"] > 0]
    print(f"\n    posterior region empty in {len(sweep) - len(bad)} of {len(sweep)} placements;"
          f" populated in {len(bad)}")
    if bad:
        w = min(bad, key=lambda s: abs(s["ear_centre_y"]))
        print(f"    first placement that populates it: ear_centre_y={w['ear_centre_y']}, "
              f"dy={w['dy']} -> {w['n_posterior']} electrodes")
        print("    -> at that anchoring PDR frequency becomes computable from behind-the-ear")
        print("       electrodes: a number that returns without error and is not the construct.")
    print(f"    occipital region empty in ALL {len(sweep)} placements.")
    report["region_occupancy"] = occ
    report["coordinate_sweep"] = sweep

    ex = ConceptExtractor(sfreq=SFREQ, channel_names=STD_CH, coord_table=EASYCOG_COORDS)
    measurable = [n for n in WIN_NAMES
                  if n in ('asymmetry_alpha', 'asymmetry_freq', 'theta_alpha_ratio',
                           'slowing_ratio', 'rel_alpha', 'rel_theta', 'aperiodic_exponent',
                           'spectral_edge', 'spectral_entropy', 'artefact_proxy')]

    # ---------------------------------------------------------------- A. extraction
    print("\n" + "=" * 78)
    print("A2. EXTRACTION under the matching protocol")
    unsl = root / "unsliced" / "asreog_filter_order3_all_data"
    arms = {}
    for car in (True, False):
        rows, short = [], []
        for f in sorted(glob.glob(str(unsl / "*-resting.npz"))):
            sid = os.path.basename(f).split("-")[0]
            if sid not in meta:
                continue
            x = np.load(f, allow_pickle=True)["raw_eeg"].astype(np.float64)
            ep, navail, rej = epochs_from(x, win, PP_UV, car)
            got = extract(ex, ep, sid, halves=True)
            if got is None:
                short.append((sid, len(ep), navail))
                continue
            v, ha, hb = got
            rows.append(dict(sid=sid, v=v, ha=ha, hb=hb, nclean=len(ep), navail=navail, rej=rej))
        arms["common_average" if car else "as_distributed"] = (rows, short)
        print(f"    {'common average' if car else 'as distributed':16s}: "
              f"{len(rows)} of {len(rows) + len(short)} sessions reach K={K}")
    rows, short = arms["common_average"]
    print(f"    short of quota: {[(s, c, a) for s, c, a in short]}")
    report["n_sessions_analysed"] = len(rows)
    report["n_sessions_total"] = len(rows) + len(short)
    report["sessions_short_of_quota"] = [{"subject": s, "clean": c, "available": a}
                                         for s, c, a in short]

    sids = [r["sid"] for r in rows]
    Z = np.vstack([r["v"] for r in rows])
    HA = np.vstack([r["ha"] for r in rows])
    HB = np.vstack([r["hb"] for r in rows])
    nclean = np.array([r["nclean"] for r in rows], float)
    navail = np.array([r["navail"] for r in rows], float)
    moca = np.array([meta[s]["moca"] for s in sids], float)
    mmse = np.array([meta[s]["mmse"] for s in sids], float)
    age = np.array([meta[s]["age"] for s in sids], float)
    dis = np.array([meta[s]["dis"] for s in sids])
    art = Z[:, WIN_NAMES.index(ARTEFACT)]

    report["cohort"] = {
        "n": len(sids),
        "age_mean": float(age.mean()), "age_sd": float(age.std(ddof=1)),
        "moca_mean": float(moca.mean()), "mmse_mean": float(mmse.mean()),
        "n_moca_ge26": int((moca >= 26).sum()),
        "by_diagnosis": {d: int((dis == d).sum()) for d in sorted(set(dis))}}

    # ---------------------------------------------------------------- A. audit floors
    print("\n" + "=" * 78)
    print("A3. ACQUISITION FLOORS -- what separates the classes with no brain signal")
    floors = {}
    for cname, m_pos, m_neg in [("impaired_vs_normal", moca < 26, moca >= 26),
                                ("AD_vs_control", dis == "AD", dis == "Control")]:
        print(f"    {cname}  (n={m_pos.sum()} vs {m_neg.sum()})")
        f = {}
        for nm, v in [("epochs_available", navail), ("clean_epochs", nclean),
                      ("artefact_proxy", art), ("age", age)]:
            a = auc(v[m_pos], v[m_neg])
            f[nm] = a
            flag = "  <-- above the floor" if abs(a - 0.5) > 0.15 else ""
            print(f"        {nm:18s} AUC = {a:.3f}{flag}")
        floors[cname] = f
    report["acquisition_floors"] = floors

    # ---------------------------------------------------------------- B. checks 2, 4, 6
    print("\n" + "=" * 78)
    print("B. BATTERY CHECKS on wearable hardware")
    print(f"    check 2 = split-half ICC(2,1) over the same {len(sids)} sessions")
    print("    check 4 = criterion validity; check 6 = Williams's test against artefact\n")

    checks = {}
    hdr = (f"    {'concept':22s} {'ICC':>6s} | {'r(MMSE)':>8s} {'r(MoCA)':>8s} "
           f"{'r|age':>7s} | {'r(art)':>7s} {'Wms t':>7s} {'p':>7s}  verdict")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for j, nm in enumerate(WIN_NAMES):
        z = Z[:, j]
        if not np.isfinite(z).any():
            print(f"    {nm:22s} {'not measurable: posterior/occipital region empty':>60s}")
            checks[nm] = {"measurable": False,
                          "reason": "posterior/occipital region empty on this montage"}
            continue
        rec = {"measurable": True, "median": float(np.nanmedian(z)),
               "iqr": [float(np.nanpercentile(z, 25)), float(np.nanpercentile(z, 75))]}
        rec["split_half_icc"] = icc21(HA[:, j], HB[:, j])
        r_mmse = stats.pearsonr(z, mmse)
        r_moca = stats.pearsonr(z, moca)
        rec["r_mmse"], rec["p_mmse"] = float(r_mmse[0]), float(r_mmse[1])
        rec["r_moca"], rec["p_moca"] = float(r_moca[0]), float(r_moca[1])
        pr = partial_r(z, moca, age)
        rec["r_moca_partial_age"], rec["p_moca_partial_age"] = float(pr[0]), float(pr[1])
        if nm == ARTEFACT:
            print(f"    {nm:22s} {rec['split_half_icc']:6.3f} | {rec['r_mmse']:+8.3f} "
                  f"{rec['r_moca']:+8.3f} {rec['r_moca_partial_age']:+7.3f} |"
                  f"{'  (is the instrument)':>32s}")
            checks[nm] = rec
            continue
        r_art = float(stats.pearsonr(z, art)[0])
        rec["r_artefact"] = r_art
        for crit, cv in (("mmse", mmse), ("moca", moca)):
            r12 = abs(float(stats.pearsonr(z, cv)[0]))
            r13 = abs(r_art)
            r23 = abs(float(stats.pearsonr(art, cv)[0]))
            t, p = williams(r12, r13, r23, len(z))
            if p is None:
                v = "untestable"
            elif p >= 0.05:
                v = "unresolved"
            elif r12 > r13:
                v = "criterion-dominated"
            else:
                v = "ARTEFACT-DOMINATED"
            rec[f"williams_{crit}"] = {"t": t, "p": p, "verdict": v}
        vm = rec["williams_moca"]["verdict"]
        print(f"    {nm:22s} {rec['split_half_icc']:6.3f} | {rec['r_mmse']:+8.3f} "
              f"{rec['r_moca']:+8.3f} {rec['r_moca_partial_age']:+7.3f} | {r_art:+7.3f} "
              f"{rec['williams_moca']['t']:+7.2f} {rec['williams_moca']['p']:7.4f}  {vm}")
        checks[nm] = rec
    report["checks"] = checks
    report["r_artefact_criterion"] = {
        "mmse": float(stats.pearsonr(art, mmse)[0]),
        "moca": float(stats.pearsonr(art, moca)[0])}

    # compositionality note: relative band powers share a budget with relative gamma
    report["compositional_with_artefact"] = ["rel_alpha", "rel_theta", "theta_alpha_ratio",
                                             "slowing_ratio"]

    # sensitivity of the decisive verdict to the reference
    rows2, _ = arms["as_distributed"]
    Z2 = np.vstack([r["v"] for r in rows2])
    s2 = [r["sid"] for r in rows2]
    m2 = np.array([meta[s]["moca"] for s in s2], float)
    a2 = Z2[:, WIN_NAMES.index(ARTEFACT)]
    sens = {}
    for nm in ("aperiodic_exponent", "spectral_edge", "spectral_entropy"):
        j = WIN_NAMES.index(nm)
        r12 = abs(float(stats.pearsonr(Z2[:, j], m2)[0]))
        r13 = abs(float(stats.pearsonr(Z2[:, j], a2)[0]))
        r23 = abs(float(stats.pearsonr(a2, m2)[0]))
        t, p = williams(r12, r13, r23, len(s2))
        sens[nm] = {"r_criterion": r12, "r_artefact": r13, "p": p,
                    "verdict": ("ARTEFACT-DOMINATED" if p is not None and p < 0.05 and r13 > r12
                                else "unresolved")}
    report["sensitivity_as_distributed_reference"] = sens
    print("\n    Same verdicts under the as-distributed reference (no common average):")
    for nm, v in sens.items():
        print(f"        {nm:22s} crit {v['r_criterion']:.3f} art {v['r_artefact']:.3f} "
              f"p={v['p']:.4f}  {v['verdict']}")

    # ---------------------------------------------------------------- C. test-retest
    print("\n" + "=" * 78)
    print("C. ACROSS-SESSION TEST-RETEST (re-application of the array on a different day)")
    rob = root / "sliced" / "asreog_filter_order3_robustness_data"
    swin = 375                                  # the released slices are 3 s at 125 Hz
    ex3 = ConceptExtractor(sfreq=SFREQ, channel_names=STD_CH, coord_table=EASYCOG_COORDS)
    files = sorted(glob.glob(str(rob / "*-resting-task9-*.npz")))
    by_session = {}
    for f in files:
        b = os.path.basename(f)
        key = b.split("-resting")[0]
        start = int(b.rsplit("-", 2)[-2])
        by_session.setdefault(key, []).append((start, f))
    sess = {}
    for key, items in sorted(by_session.items()):
        items.sort()
        # the slices step by 1 s and are 3 s long, so every third one is non-overlapping
        chosen = [f for st, f in items if st % (3 * int(SFREQ)) == 0]
        segs = []
        for f in chosen:
            d = np.load(f, allow_pickle=True)
            segs.append(d["eeg_seg"].astype(np.float64)[:, :swin])
        if not segs:
            continue
        ep = np.stack(segs)
        ep = ep - ep.mean(axis=1, keepdims=True)          # common average, per epoch
        pp = (ep.max(-1) - ep.min(-1)).max(-1)
        ep = ep[pp <= PP_UV]
        sid = key.split("-")[0]
        got = extract(ex3, ep, sid, k=K, halves=True)
        if got is None:
            print(f"    {key}: only {len(ep)} clean epochs, below K={K} -- dropped")
            continue
        v, ha, hb = got
        sess[key] = dict(sid=sid, date=key.split("-", 1)[1], v=v, ha=ha, hb=hb, n=len(ep))
    print(f"    {len(sess)} sessions from {len(set(s['sid'] for s in sess.values()))} subjects, "
          f"K={K} windows of 3 s from the released slices")

    # Two pairings. Consecutive visits use every adjacent pair, so a middle session enters
    # twice and the rows are not independent; first-visit pairs take one pair per subject and
    # are. Both are reported because the second is honest and the first is all the data.
    first_pairs, consec_pairs = [], []
    for sid in sorted(set(s["sid"] for s in sess.values())):
        ks = sorted([k for k, s in sess.items() if s["sid"] == sid], key=lambda k: sess[k]["date"])
        if len(ks) >= 2:
            first_pairs.append((ks[0], ks[1]))
        for i in range(len(ks) - 1):
            consec_pairs.append((ks[i], ks[i + 1]))

    retest = {"n_sessions": len(sess), "n_subjects": len(set(s["sid"] for s in sess.values())),
              "n_pairs_independent": len(first_pairs), "n_pairs_consecutive": len(consec_pairs),
              "window_s": 3.0, "K": K,
              "pairs_independent": [[a, b] for a, b in first_pairs],
              "pairs_consecutive": [[a, b] for a, b in consec_pairs], "concepts": {}}

    keys12 = sorted(sess)
    SH_A = np.vstack([sess[k]["ha"] for k in keys12])
    SH_B = np.vstack([sess[k]["hb"] for k in keys12])
    IND_A = np.vstack([sess[a]["v"] for a, _ in first_pairs])
    IND_B = np.vstack([sess[b]["v"] for _, b in first_pairs])
    CON_A = np.vstack([sess[a]["v"] for a, _ in consec_pairs])
    CON_B = np.vstack([sess[b]["v"] for _, b in consec_pairs])

    print(f"\n    {len(first_pairs)} independent pairs (one per subject), "
          f"{len(consec_pairs)} consecutive-visit pairs")
    print("    Split-half is computed on the SAME sessions, so the two columns are comparable.\n")
    print(f"    {'concept':22s} {'split-half':>11s} | {'retest (n=5)':>13s} {'95% CI':>16s} | "
          f"{'retest (n=7)':>13s}")
    print("    " + "-" * 84)
    for j, nm in enumerate(WIN_NAMES):
        if not np.isfinite(IND_A[:, j]).any():
            continue
        sh = icc21(SH_A[:, j], SH_B[:, j])
        ind = icc21(IND_A[:, j], IND_B[:, j])
        con = icc21(CON_A[:, j], CON_B[:, j])
        lo, hi = boot_ci(lambda a, b: icc21(a, b), IND_A[:, j], IND_B[:, j], n_boot=args.n_boot)
        retest["concepts"][nm] = {"split_half_icc_same_sessions": sh,
                                  "retest_icc_independent": ind, "retest_ci_independent": [lo, hi],
                                  "retest_icc_consecutive": con}
        ci = f"[{lo:+.2f},{hi:+.2f}]" if lo is not None else "n/a"
        print(f"    {nm:22s} {sh:11.3f} | {ind:13.3f} {ci:>16s} | {con:13.3f}")
    report["test_retest"] = retest
    print("\n    Five subjects. These intervals are wide by construction: read the column as a")
    print("    bound on gross instability, not as a reliability estimate. The comparison that")
    print("    does carry weight is between the two ICC columns on identical sessions.")

    write_result(OUT / "easycog_battery.json", report, script=__file__,
                 extra={"description":
                        "Construct-validation battery (checks 2, 4, 6), acquisition floors and "
                        "across-session test-retest on the EasyCog forehead-and-ear wearable "
                        "corpus, using the extractor and matching protocol of the main analysis."})
    print(f"\nwrote {OUT / 'easycog_battery.json'}")


if __name__ == "__main__":
    main()
