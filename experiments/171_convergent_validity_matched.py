"""Check 5 of the battery -- convergent validity -- on the matched corpus.

Check 5 compares each concept against an independently written implementation of the same
quantity and detects coding error only. It cannot falsify a construct: two implementations of the
wrong formula agree perfectly.

The figure the manuscript quoted for it (rank correlations 0.81 to 0.99) came from
`outputs/results/validity/concept_validity.json`, which was computed on 1{,}529 recordings before
harmonisation and carries no provenance block. Every other check in the battery is computed on the
matched corpus of 1{,}296. One check reported from a different corpus, in a paper whose argument is
that provenance and corpus discipline decide what a number means, is not defensible even when the
check is the least consequential of the seven.

This recomputes it on `harmonised_features/channel_clean`, which is the corpus every other reported
result uses. Nothing is re-extracted: the two implementations already sit side by side in that
directory.

  * `window_concepts.npy`  -- `ConceptExtractor`, torch, the estimators the model consumes
  * `spectral.npy`         -- `window_features.spectral_stream`, SciPy Welch, written separately
                              for the baseline models and never used by the concept path

The spectral block is 123-d: 19 channels x 5 bands (0-94), 5 regions x 5 bands (95-119), then
[theta/alpha ratio, individual alpha peak frequency, spectral entropy] (120-122).

    python experiments/171_convergent_validity_matched.py --tag channel_clean
Out: outputs/results/audit/convergent_validity_<tag>.json
"""
from __future__ import annotations

import argparse
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

N_CH, N_BAND = 19, 5
BAND = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}
TAR, IAPF, SENT = 120, 121, 122


def chan_band_mean(S, band):
    """Mean relative power in one band across the 19 per-channel entries."""
    cols = [c * N_BAND + BAND[band] for c in range(N_CH)]
    return S[:, cols].mean(axis=1)


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
    check_table(d)                       # both arrays join by row index
    C_win = np.load(d / "window_concepts.npy")
    S_win = np.load(d / "spectral.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    if len(S_win) != len(C_win):
        raise SystemExit(f"row mismatch: {len(C_win)} concepts vs {len(S_win)} spectral")

    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    # Aggregate both sides identically -- the concept side through the same function the model
    # uses, the spectral side by the median it would receive.
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    S = np.stack([np.nanmedian(S_win[i], axis=0) for i in idxs])

    ci = {n: i for i, n in enumerate(CONCEPT_NAMES)}
    pairs = {
        "pdr_frequency": ("individual alpha peak frequency (all-channel argmax, SciPy Welch)",
                          S[:, IAPF]),
        "theta_alpha_ratio": ("theta/alpha from independent band integration", S[:, TAR]),
        "rel_theta": ("mean relative theta over 19 channels", chan_band_mean(S, "theta")),
        "rel_alpha": ("mean relative alpha over 19 channels", chan_band_mean(S, "alpha")),
        "artefact_proxy": ("mean relative gamma over 19 channels", chan_band_mean(S, "gamma")),
        "spectral_entropy": ("spectral entropy (full-band, SciPy)", S[:, SENT]),
    }

    report = {
        "note": "Check 5 (convergent validity) recomputed on the matched corpus. Supersedes the "
                "values in outputs/results/validity/concept_validity.json, which were computed "
                "on 1,529 pre-harmonisation recordings and carry no provenance block.",
        "tag": args.tag,
        "n_subjects": int(len(subs)),
        "n_windows": int(len(groups)),
        "concepts": {},
    }
    print(f"corpus {args.tag}: {len(subs)} subjects, {len(groups):,} windows\n")
    print(f"{'concept':22s} {'rho':>8s}   compared against")
    rhos = []
    for name, (against, ref) in pairs.items():
        r = spearman(C[:, ci[name]], ref)
        rhos.append(r)
        report["concepts"][name] = {"against": against, "spearman": r}
        print(f"{name:22s} {r:8.4f}   {against}")

    lo, hi = min(rhos), max(rhos)
    report["range"] = [lo, hi]
    report["interpretation"] = (
        "Check 5 detects coding error only; agreement between two implementations of the same "
        "formula is not evidence that the formula measures its construct.")
    print(f"\nrange {lo:.3f} to {hi:.3f} across {len(rhos)} concepts")

    dest = write_result(OUT / f"convergent_validity_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
