"""R1: is the "muscle-artefact proxy" actually measuring muscle?

The claim the manuscript rests on is that three named concepts track scalp electromyogram. The
evidence offered is their correlation with `artefact_proxy`, which is the mean relative 30-45 Hz
power over 19 electrodes -- computed from the same Welch spectrum as the concepts themselves
(`src/models/v6/concepts.py:193`). Two problems follow, and neither is answerable by looking at
that correlation again:

1. **Part of the correlation is algebraic.** A 95th-percentile spectral edge over 1-30 Hz, a
   full-band spectral entropy and a 2-40 Hz log-log slope are deterministic functionals of the
   same power spectral density as relative gamma. They *must* share variance with it. Observing
   that they do is not evidence about muscle.
2. **The proxy's own convergent check is circular.** `112_concept_validity.py:78` validates
   `artefact_proxy` against "mean relative gamma over 19 channels" -- a reimplementation of
   itself. Agreement of rho=0.98 measures code, not construct.

So this script asks the question the correlation cannot: does the high-frequency power that these
concepts track carry the *signature* of muscle, or the signature of something else?

--------------------------------------------------------------------------------------------
The physiological prior, and why it discriminates
--------------------------------------------------------------------------------------------

Scalp EMG has a known topography. Whitham et al. (2007, Clin Neurophysiol 118:1877) recorded EEG
during complete neuromuscular blockade and found power above 20 Hz fell by one to two orders of
magnitude; in the unparalysed state that power was "most evident in lateral and posterior leads,
consistent with a localisation close to cranial and cervical muscles". Goncharova et al. (2003,
Clin Neurophysiol 114:1580) mapped the same thing directly: frontalis and temporalis dominate,
contamination is greatest at temporal and inferior-frontal electrodes, and least at the vertex.

That gives predictions which separate muscle from the alternatives:

  P1  TOPOGRAPHY. Gamma-band power is larger at temporalis/frontalis-adjacent electrodes
      (T3, T4, F7, F8) than at midline electrodes (Fz, Cz, Pz). Physiological gamma and
      wideband electrical noise carry no such asymmetry.

  P2  COMPOSITION. If the concepts track muscle, they should track the LATERAL gamma more
      strongly than the MIDLINE gamma, because only the lateral signal is muscle-loaded. If
      they track a global spectral property, the two should be interchangeable.

  P3  BROADBAND. EMG is broadband. Subjects with high gamma should show elevated beta too, and
      the lateral-minus-midline contrast should agree between the two bands. Narrowband
      physiological gamma would not do this.

  P4  SPECIFICITY. Concepts that PASSED the original audit (posterior rhythm frequency,
      theta/alpha) should show none of the above. If they show the same pattern, the pattern is
      an artefact of the analysis rather than of muscle.

P2 is the load-bearing test and it is not circular: the concepts are computed over all 19
electrodes, so nothing forces them to prefer the lateral subset.

--------------------------------------------------------------------------------------------
What this can and cannot establish
--------------------------------------------------------------------------------------------

CAN: whether the high-frequency variance the concepts share carries a myogenic topography.
CANNOT: prove any individual recording is contaminated. That needs an EMG channel or an
ICA muscle-component labelling, neither of which this corpus supports today. The honest verdict
is therefore "consistent with EMG" or "not consistent with EMG", never "is EMG".

One measurement caveat is stated in the output because it changes interpretation: windows are
per-channel z-scored (`src/data/dataset_wrapper.py:353`), so every channel's TOTAL power is
normalised to a constant. Relative band powers are ratios within a channel and therefore survive
this untouched -- which is what makes the topographic test valid -- but it means the test is on
the topography of spectral SHAPE (what fraction of a channel's power is above 30 Hz), not of
amplitude. For EMG that is the right quantity: muscle adds broadband power at the electrodes
nearest the muscle, which raises the high-frequency fraction there.

Correlations are computed WITHIN cohort and reported per cohort, never pooled, because pooling
eight cohorts that differ in both artefact burden and concept distribution produces an ecological
correlation that says nothing about individuals.

    python experiments/117_artefact_proxy_validation.py
Out: outputs/results/validity/artefact_proxy_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _audit_cache import load_corpus, subject_table  # noqa: E402
from _protocol_v2 import CN, DEM, MCI  # noqa: E402

OUT = ROOT / "outputs" / "results" / "validity"
TA = ROOT / "outputs" / "results" / "track_a"
SPECTRAL_SLICE = slice(955, 1078)
W_EMGFREE = ROOT / "outputs" / "results" / "validity" / "window_concepts_emgfree.npy"
W_ORIGINAL = ROOT / "outputs" / "results" / "reliability" / "window_concepts.npy"

#: Channel order used throughout this repository (src/config/config.py:70).
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

#: Temporalis / inferior-frontalis adjacent. T5 and T6 are DELIBERATELY EXCLUDED even though they
#: are the most muscle-loaded electrodes of all: GENEEG supplies only 17 of 19 channels and the
#: loader zero-fills exactly these two (`_protocol_v2.DROP_CHANNELS = [14, 15]`), so their band
#: powers are undefined there. Including them would make the contrast incomparable across the one
#: cohort the paper most needs it for.
LATERAL = ['T3', 'T4', 'F7', 'F8']
#: Furthest from cranial and cervical musculature; the vertex is the classical EMG-free reference.
MIDLINE = ['Fz', 'Cz', 'Pz']

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
B_BETA, B_GAMMA = 3, 4

#: Concepts the original audit judged artefact-dominated, and the ones it cleared. The cleared
#: set is the specificity control (P4): whatever pattern the failed set shows, these must not.
FAILED = ["aperiodic_exponent", "spectral_entropy", "spectral_edge"]
PASSED = ["pdr_frequency", "theta_alpha_ratio", "rel_alpha", "rel_theta"]


# ------------------------------------------------------------------ statistics

def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_ci(a, b, n_boot=2000, seed=42):
    """Point estimate with a subject-level bootstrap percentile interval."""
    r = spearman(a, b)
    if r is None:
        return None, None, None
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        v = spearman(a[i], b[i])
        if v is not None:
            boots.append(v)
    if len(boots) < 100:
        return r, None, None
    return r, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z, via residuals of the rank variables.

    Rank-transform all three, then regress out z linearly. This is the standard Spearman partial
    and it is what P2 needs: whether a concept still tracks lateral gamma once the globally
    shared high-frequency level (midline gamma) has been removed, and vice versa.
    """
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 15:
        return None
    rx, ry, rz = _rank(x[ok]), _rank(y[ok]), _rank(z[ok])
    rz = (rz - rz.mean()) / (rz.std() + 1e-12)

    def resid(v):
        v = v - v.mean()
        return v - float(v @ rz / (rz @ rz + 1e-12)) * rz

    ex, ey = resid(rx), resid(ry)
    if ex.std() == 0 or ey.std() == 0:
        return None
    return float(np.corrcoef(ex, ey)[0, 1])


def wilcoxon_signed_rank(d, n_perm=10000, seed=42):
    """Two-sided sign-flip permutation test on paired differences, plus a rank-biserial size.

    A permutation test rather than a normal approximation because n varies from 63 to 1120 here
    and the differences are heavily skewed.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 10:
        return None, None, None
    rng = np.random.default_rng(seed)
    r = _rank(np.abs(d)) + 1.0
    w_pos = float(r[d > 0].sum())
    w_neg = float(r[d < 0].sum())
    obs = w_pos - w_neg
    null = np.array([float((r * rng.choice([-1.0, 1.0], len(d))).sum())
                     for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(obs)).mean())
    rbc = (w_pos - w_neg) / (w_pos + w_neg)          # rank-biserial correlation, [-1, 1]
    return float(np.median(d)), rbc, max(p, 1.0 / n_perm)


def fmt(v, nd=3):
    return " n/a " if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=40,
                    help="smallest cohort for which correlations are reported at all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import CONCEPT_NAMES

    c = load_corpus()
    st = subject_table(c)
    subs, idxs = st["subjects"], st["index"]
    sub_coh, sub_y3 = st["cohort"], st["y3"]
    print(f"{len(subs)} subjects | corpus {c['meta']['n_windows']:,} windows "
          f"(augmentation off)\n")

    # ---- per-channel relative band powers, subject level ---------------------------------
    # X_full[:, 955:1078] layout (window_features.spectral_stream:70):
    #   0-94    19 channels x 5 bands
    #   95-119  5 regions x 5 bands
    #   120-122 theta/alpha, IAPF, spectral entropy
    Xs = np.load(TA / "X_full.npy", mmap_mode="r")[:, SPECTRAL_SLICE]
    ci = {ch: i for i, ch in enumerate(CHANNELS)}

    def chan_band(band):
        """[n_subjects, 19] subject-median relative power in `band` per channel."""
        cols = [i * 5 + band for i in range(19)]
        return np.stack([np.median(np.asarray(Xs[ix][:, cols], dtype=np.float64), axis=0)
                         for ix in idxs])

    G = chan_band(B_GAMMA)
    B = chan_band(B_BETA)

    # Sanity: GENEEG zero-fills T5/T6. If that is true here, those columns are degenerate and
    # their exclusion from LATERAL is not a preference but a requirement.
    gen = sub_coh == "GENEEG"
    if gen.any():
        z = {ch: float(np.median(G[gen, ci[ch]])) for ch in ("T5", "T6", "T3", "Pz")}
        print(f"GENEEG zero-fill check, median relative gamma: {z}")
        print("  (T5/T6 are excluded from the lateral set for this reason)\n")

    lat = [ci[ch] for ch in LATERAL]
    mid = [ci[ch] for ch in MIDLINE]
    g_lat, g_mid = G[:, lat].mean(1), G[:, mid].mean(1)
    b_lat, b_mid = B[:, lat].mean(1), B[:, mid].mean(1)
    g_all = G.mean(1)                       # the manuscript's artefact_proxy, reconstructed
    emg_index = g_lat - g_mid               # the topographic EMG signature

    # ---- concepts ------------------------------------------------------------------------
    W = np.load(W_EMGFREE if W_EMGFREE.exists() else W_ORIGINAL, mmap_mode="r")
    W0 = np.load(W_ORIGINAL, mmap_mode="r")
    j = {n: k for k, n in enumerate(CONCEPT_NAMES)}
    # aggregate_subject() maps 15 window concepts -> 17 subject concepts; for the three concepts
    # under test the mapping is a plain median, so take it directly and avoid re-deriving it.
    WIN_COL = {"aperiodic_exponent": 11, "spectral_edge": 12, "spectral_entropy": 13,
               "pdr_frequency": 0, "theta_alpha_ratio": 7, "rel_alpha": 9, "rel_theta": 10}

    def concept(name, unrepaired=False):
        src = W0 if unrepaired else W
        col = WIN_COL[name]
        return np.array([float(np.nanmedian(np.asarray(src[ix][:, col], dtype=np.float64)))
                         for ix in idxs])

    con = {n: concept(n) for n in FAILED + PASSED}
    con_pre = {n: concept(n, unrepaired=True) for n in FAILED}

    # Reconstruction check: g_all should reproduce the stored artefact_proxy closely. It is a
    # different estimator (SciPy Welch, 25% overlap) over the same band, so agreement is
    # convergent validity for the RECONSTRUCTION only -- it says nothing about the construct.
    stored_proxy = np.array([float(np.nanmedian(np.asarray(W[ix][:, 14], dtype=np.float64)))
                             for ix in idxs])
    r_rec = spearman(g_all, stored_proxy)
    print(f"reconstruction of artefact_proxy from the 123-d block: rho = {fmt(r_rec)}")
    print("  (this is code agreement, not construct validity -- see docstring)\n")

    report = {
        "note": "R1. Does the high-frequency variance the failed concepts track carry a myogenic "
                "topography? Predictions from Whitham 2007 and Goncharova 2003.",
        "lateral_channels": LATERAL, "midline_channels": MIDLINE,
        "measurement_caveat": "windows are per-channel z-scored, so this tests the topography of "
                              "spectral shape (high-frequency fraction), not of amplitude",
        "proxy_reconstruction_rho": r_rec,
        "cohorts": {},
    }

    cohorts = [ch for ch in np.unique(sub_coh) if (sub_coh == ch).sum() >= args.min_n]
    for ch in cohorts:
        m = sub_coh == ch
        n = int(m.sum())
        row = {"n": n}
        print("=" * 78)
        print(f"{ch}  (n={n})")

        # ---- P1 topography -------------------------------------------------------------
        med, rbc, p = wilcoxon_signed_rank(g_lat[m] - g_mid[m])
        medb, rbcb, pb = wilcoxon_signed_rank(b_lat[m] - b_mid[m])
        row["P1_topography"] = {
            "gamma_lateral_median": float(np.median(g_lat[m])),
            "gamma_midline_median": float(np.median(g_mid[m])),
            "paired_median_diff": med, "rank_biserial": rbc, "p": p,
            "beta_paired_median_diff": medb, "beta_rank_biserial": rbcb, "beta_p": pb,
            "verdict": ("lateral > midline (consistent with EMG)" if rbc and rbc > 0.3 and p < 0.05
                        else "no lateral excess (NOT consistent with EMG)"),
        }
        print(f"  P1 gamma lateral {np.median(g_lat[m]):.4f} vs midline "
              f"{np.median(g_mid[m]):.4f} | rank-biserial {fmt(rbc)} p={p:.4f}")
        print(f"     beta  same contrast: rank-biserial {fmt(rbcb)} p={pb:.4f}")

        # ---- P2 composition -------------------------------------------------------------
        print(f"  P2 does each concept prefer LATERAL gamma over MIDLINE gamma?")
        print(f"     {'concept':22s} {'rho_lat':>8s} {'rho_mid':>8s} "
              f"{'part.lat|mid':>13s} {'part.mid|lat':>13s}")
        p2 = {}
        for name in FAILED + PASSED:
            v = con[name][m]
            rl, rl_lo, rl_hi = spearman_ci(v, g_lat[m])
            rm, rm_lo, rm_hi = spearman_ci(v, g_mid[m])
            plm = partial_spearman(v, g_lat[m], g_mid[m])
            pml = partial_spearman(v, g_mid[m], g_lat[m])
            p2[name] = {"rho_lateral": rl, "rho_lateral_ci": [rl_lo, rl_hi],
                        "rho_midline": rm, "rho_midline_ci": [rm_lo, rm_hi],
                        "partial_lateral_given_midline": plm,
                        "partial_midline_given_lateral": pml,
                        "rho_emg_index": spearman(v, emg_index[m]),
                        "group": "failed" if name in FAILED else "passed"}
            print(f"     {name:22s} {fmt(rl):>8s} {fmt(rm):>8s} "
                  f"{fmt(plm):>13s} {fmt(pml):>13s}")
        row["P2_composition"] = p2

        # ---- P3 broadband ---------------------------------------------------------------
        r_gb = spearman(g_all[m], B[m].mean(1))
        r_contrast = spearman(g_lat[m] - g_mid[m], b_lat[m] - b_mid[m])
        row["P3_broadband"] = {
            "rho_gamma_beta_global": r_gb,
            "rho_lateral_contrast_gamma_vs_beta": r_contrast,
            "verdict": ("broadband, topographically coherent (consistent with EMG)"
                        if (r_gb or 0) > 0.3 and (r_contrast or 0) > 0.3
                        else "not broadband/coherent"),
        }
        print(f"  P3 rho(gamma, beta) = {fmt(r_gb)} | "
              f"rho(lateral contrast: gamma vs beta) = {fmt(r_contrast)}")

        # ---- P4 specificity, summarised --------------------------------------------------
        def mean_abs(names, key):
            v = [abs(p2[n][key]) for n in names if p2[n][key] is not None]
            return float(np.mean(v)) if v else None

        row["P4_specificity"] = {
            "failed_mean_abs_partial_lat": mean_abs(FAILED, "partial_lateral_given_midline"),
            "passed_mean_abs_partial_lat": mean_abs(PASSED, "partial_lateral_given_midline"),
        }
        f_, p_ = (row["P4_specificity"]["failed_mean_abs_partial_lat"],
                  row["P4_specificity"]["passed_mean_abs_partial_lat"])
        print(f"  P4 mean |partial lat|mid|: failed {fmt(f_)}  passed {fmt(p_)}")

        # ---- did the repair reduce the myogenic coupling? --------------------------------
        rep = {}
        for name in FAILED:
            rep[name] = {
                "emg_index_before": spearman(con_pre[name][m], emg_index[m]),
                "emg_index_after": spearman(con[name][m], emg_index[m]),
            }
        row["repair_effect_on_emg_index"] = rep
        print("  repair effect, rho with the topographic EMG index:")
        for name, d in rep.items():
            print(f"     {name:22s} {fmt(d['emg_index_before'])} -> "
                  f"{fmt(d['emg_index_after'])}")

        # ---- does the proxy differ by diagnosis here? ------------------------------------
        by_class = {}
        for lab, nm in ((CN, "CN"), (MCI, "MCI"), (DEM, "dementia")):
            k = m & (sub_y3 == lab)
            if k.sum() >= 8:
                by_class[nm] = {"n": int(k.sum()),
                                "gamma_all_median": float(np.median(g_all[k])),
                                "emg_index_median": float(np.median(emg_index[k]))}
        row["by_class"] = by_class
        print(f"  by class: " + "  ".join(
            f"{k} n={v['n']} gamma={v['gamma_all_median']:.4f} emg={v['emg_index_median']:+.4f}"
            for k, v in by_class.items()))
        print(flush=True)

        report["cohorts"][ch] = row

    dest = OUT / "artefact_proxy_validation.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print("=" * 78)
    print(f"wrote {dest}")
    print("\nHOW TO READ THIS. The proxy is consistent with EMG only if, in the training cohort "
          "and\nat the external sites alike: P1 shows a lateral excess, P2 shows the FAILED "
          "concepts\ntracking lateral gamma after midline gamma is partialled out (and not the "
          "reverse),\nP3 shows a coherent broadband rise, and P4 shows the PASSED concepts doing "
          "none of it.\nAnything less should be reported as 'high-frequency broadband power of "
          "undetermined\norigin', which is a weaker but defensible claim.")


if __name__ == "__main__":
    main()
