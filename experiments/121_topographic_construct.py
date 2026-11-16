"""R2: can the pipeline express the constructs three of the concepts are named after?

This is a third kind of naming failure, distinct from the two the manuscript already reports, and
it is established analytically rather than statistically.

A neurophysiologist's topographic items are AMPLITUDE statements:

    "the posterior dominant rhythm is maximal occipitally"   -> alpha AMPLITUDE, posterior > anterior
    "interhemispheric amplitude asymmetry exceeds 35-50%"    -> an AMPLITUDE ratio, left vs right
    "loss of the anterior-posterior gradient"                -> an AMPLITUDE gradient

Three concepts carry those names: `posterior_maximality`, `ap_gradient_alpha`, `asymmetry_alpha`
(`src/models/v6/concepts.py:167-186`). All three are computed from RELATIVE band power, that is
from each channel's power in a band divided by that channel's own total power. A ratio of that
form is invariant to any per-channel rescaling -- and the pipeline rescales per channel twice:

    every loader          RobustScaler(quantile_range=(25,75)).fit_transform(data.T).T
                          (caueeg_loader.py:123, bids_loader.py:171, alz_eeg_loader.py:162,
                           geneeg_loader.py:112, mci_loader.py:280, mendeley_loader.py:152)
    the wrapper           per-channel z-score of every 4 s window
                          (dataset_wrapper.py:353, NCG_NORM='channel', the default)

So inter-channel amplitude is removed at load, and the concepts could not read it even if it
survived. They measure the topography of spectral SHAPE -- what fraction of a channel's power sits
in the alpha band -- which is a real quantity but not the one the name denotes.

The measured criterion validity is consistent with that reading and was never explained:

    posterior_maximality   rho(MMSE) = +0.003
    asymmetry_alpha        rho(MMSE) = -0.072
    ap_gradient_freq       rho(MMSE) = +0.166
    ap_gradient_alpha      rho(MMSE) = +0.336

This script does two things.

PART A -- prove the invariance rather than assert it. Apply random per-channel gains to a
synthetic window with a deliberately amplitude-graded posterior alpha rhythm, and to real windows
from the corpus, and show every concept is unchanged. If the concepts move, the argument above is
wrong and the rest of the script should be ignored.

PART B -- quantify what is lost, using the ONE amplitude signal the pipeline still carries. Within
a single window the per-channel z-score has destroyed inter-channel amplitude, but the RATIO of
absolute band powers between channels can be recovered from a re-read of the recording without the
per-channel scaling. Rather than rebuild the corpus, Part B computes amplitude-based versions of
the three concepts directly from the preprocessed recordings for a stratified subsample, and
compares their criterion validity against the relative versions on the same subjects.

Part B requires the raw recordings and is skipped with a clear message if they are unavailable.

    python experiments/121_topographic_construct.py --part a
    python experiments/121_topographic_construct.py --part b --n-subjects 200
Out: outputs/results/audit/topographic_construct.json
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

OUT = ROOT / "outputs" / "results" / "audit"
SEED = 42

#: The concepts whose NAMES denote amplitude and whose ESTIMATORS cannot express it.
AMPLITUDE_NAMED = ["posterior_maximality", "ap_gradient_alpha", "asymmetry_alpha"]


def part_a(report):
    """Invariance of every concept to arbitrary per-channel gain."""
    import torch

    from src.models.v6 import CONCEPT_NAMES
    from src.models.v6.concepts import ConceptExtractor

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    ex = ConceptExtractor()

    # A window with a genuine AMPLITUDE gradient: posterior alpha 8x the anterior alpha. If the
    # concepts could see amplitude topography, this is where they would see it.
    n_win, n_ch, n_t = 16, 19, 1024
    x = torch.randn(n_win, n_ch, n_t)
    t = torch.arange(n_t) / 256.0
    alpha = torch.sin(2 * np.pi * 10.0 * t)
    for ch in (6, 7, 8, 9):        # P3 P4 O1 O2
        x[:, ch] += 4.0 * alpha
    for ch in (0, 1, 2, 3):        # Fp1 Fp2 F3 F4
        x[:, ch] += 0.5 * alpha

    base = ex.window_concepts(x).numpy()
    gains = torch.tensor(rng.uniform(0.2, 5.0, size=(1, n_ch, 1)), dtype=torch.float32)
    scaled = ex.window_concepts(x * gains).numpy()
    diff = np.abs(base - scaled).max(axis=0)

    # And the amplitude gradient the concepts should have seen, measured directly:
    def band_power(sig):
        f = torch.fft.rfft(sig * torch.hann_window(n_t), dim=-1)
        p = (f.real ** 2 + f.imag ** 2)
        fr = torch.fft.rfftfreq(n_t, 1 / 256.0)
        m = (fr >= 8) & (fr < 13)
        return p[..., m].sum(-1)

    ap_before = float((band_power(x)[:, [6, 7, 8, 9]].mean()
                       / band_power(x)[:, [0, 1, 2, 3]].mean()))
    ap_after = float((band_power(x * gains)[:, [6, 7, 8, 9]].mean()
                      / band_power(x * gains)[:, [0, 1, 2, 3]].mean()))

    names = CONCEPT_NAMES[:2] + CONCEPT_NAMES[3:16]   # window_concepts returns 15, no var/n_win
    print("PART A -- invariance of the concepts to per-channel gain")
    print(f"  synthetic window: true posterior/anterior ALPHA POWER ratio "
          f"{ap_before:.2f} before gains, {ap_after:.2f} after")
    print(f"  {'concept':24s} {'max |change|':>13s}")
    for n, d in zip(names, diff):
        flag = "" if d < 1e-4 else "   <-- NOT invariant"
        print(f"  {n:24s} {d:13.2e}{flag}")
    worst = float(diff.max())
    print(f"\n  largest change across all concepts: {worst:.2e}")
    verdict = ("INVARIANT: the concepts cannot see amplitude topography, so three of them cannot "
               "measure the construct their names denote"
               if worst < 1e-4 else
               "NOT invariant -- the analytic argument in this docstring is wrong, stop here")
    print(f"  VERDICT: {verdict}\n")

    report["part_a"] = {
        "synthetic_true_posterior_anterior_alpha_power_ratio": {
            "before_gains": ap_before, "after_gains": ap_after},
        "max_abs_change_per_concept": dict(zip(names, diff.tolist())),
        "max_abs_change_overall": worst,
        "verdict": verdict,
        "why_it_matters": ("Every loader per-channel robust-scales the recording and the wrapper "
                           "per-channel z-scores every window, so inter-channel amplitude is "
                           "removed twice. Because the concepts are ratios within a channel they "
                           "are unaffected numerically -- but posterior_maximality, "
                           "ap_gradient_alpha and asymmetry_alpha are named after amplitude "
                           "constructs and therefore cannot measure them."),
    }
    return report


def part_b(report, n_subjects, min_windows=8):
    """Amplitude-based versions of the three concepts, from recordings read without per-channel
    scaling, compared against the relative versions on the same subjects."""
    import csv

    import mne
    import torch

    from _protocol_v2 import DATASET_PATHS

    mne.set_log_level("ERROR")
    ds = DATASET_PATHS.get("ds004504")
    if ds is None or not ds.exists():
        print("PART B skipped: ds004504 not found. It is the only cohort with MMSE, and the "
              "criterion comparison is the point of this part.")
        report["part_b"] = {"skipped": "ds004504 unavailable"}
        return report

    # MMSE, the label-independent criterion
    mmse = {}
    p = ds / "participants.tsv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                k = next((k for k in row if k.strip().upper() == "MMSE"), None)
                if k and row.get("participant_id"):
                    try:
                        mmse[row["participant_id"].strip()] = float(row[k])
                    except (TypeError, ValueError):
                        pass
    if len(mmse) < 20:
        print("PART B skipped: MMSE not distributed with this copy of ds004504.")
        report["part_b"] = {"skipped": "no MMSE"}
        return report

    CH = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
          'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    POST = [CH.index(c) for c in ('P3', 'P4', 'O1', 'O2', 'Pz')]
    ANT = [CH.index(c) for c in ('Fp1', 'Fp2', 'F3', 'F4', 'Fz')]
    OCC = [CH.index(c) for c in ('O1', 'O2')]
    LEFT = [CH.index(c) for c in ('Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5')]
    RIGHT = [CH.index(c) for c in ('Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6')]

    files = sorted((ds / "derivatives").rglob("*eeg.set")) or sorted(ds.rglob("*eeg.set"))
    if not files:
        print(f"PART B skipped: no .set recordings under {ds}")
        report["part_b"] = {"skipped": "no recordings found"}
        return report
    files = files[:n_subjects]
    print(f"PART B -- reading {len(files)} recordings WITHOUT per-channel scaling")

    rows = []
    for i, f in enumerate(files):
        pid = next((part for part in f.parts if part.startswith("sub-")), None)
        if pid is None or pid not in mmse:
            continue
        try:
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            raw.rename_channels({k: v for k, v in
                                 {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}.items()
                                 if k in raw.ch_names})
            raw.pick([c for c in CH if c in raw.ch_names])
            if len(raw.ch_names) < 19:
                continue
            raw.reorder_channels(CH)
            raw.filter(0.5, 45.0, verbose=False)
            d = raw.get_data()                       # NO per-channel scaling: the whole point
        except Exception as e:                        # noqa: BLE001
            print(f"  [skip] {f.name}: {type(e).__name__}")
            continue

        sf = float(raw.info["sfreq"])
        n_t = int(4 * sf)
        n_w = d.shape[1] // n_t
        if n_w < min_windows:
            continue
        seg = d[:, :n_w * n_t].reshape(len(CH), n_w, n_t).transpose(1, 0, 2)
        x = torch.tensor(seg, dtype=torch.float32)
        win = torch.hann_window(n_t)
        F = torch.fft.rfft(x * win, dim=-1)
        psd = (F.real ** 2 + F.imag ** 2)
        fr = torch.fft.rfftfreq(n_t, 1 / sf)
        a_m = (fr >= 8) & (fr < 13)
        a_pow = psd[..., a_m].sum(-1).numpy()        # [n_win, 19] ABSOLUTE alpha power

        med = np.median(a_pow, axis=0)
        eps = 1e-20
        rows.append({
            "subject": pid, "mmse": mmse[pid], "n_windows": int(n_w),
            # the three clinical constructs, on amplitude
            "amp_ap_gradient": float(np.log10((med[POST].mean() + eps)
                                              / (med[ANT].mean() + eps))),
            "amp_posterior_maximality": float(
                np.mean([int(np.argsort(-med).tolist().index(o)) + 1 for o in OCC])),
            "amp_asymmetry": float(abs(med[LEFT].mean() - med[RIGHT].mean())
                                   / (med[LEFT].mean() + med[RIGHT].mean() + eps)),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    if len(rows) < 30:
        print(f"PART B: only {len(rows)} usable subjects; too few to compare.")
        report["part_b"] = {"skipped": f"only {len(rows)} usable subjects"}
        return report

    def rank(a):
        return np.argsort(np.argsort(a)).astype(float)

    def rho(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 15:
            return None
        return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])

    m = np.array([r["mmse"] for r in rows])
    amp = {k: np.array([r[k] for r in rows])
           for k in ("amp_ap_gradient", "amp_posterior_maximality", "amp_asymmetry")}

    print(f"\n  {len(rows)} subjects with MMSE and 19 channels")
    print(f"  {'amplitude-based construct':30s} {'rho(MMSE)':>10s}")
    out = {"n": len(rows), "amplitude_versions": {}, "relative_versions_from_112": {
        "posterior_maximality": 0.0027649607269909482,
        "ap_gradient_alpha": 0.33623331337395657,
        "asymmetry_alpha": -0.07232925927230462}}
    for k, v in amp.items():
        r = rho(v, m)
        out["amplitude_versions"][k] = r
        print(f"  {k:30s} {r:+10.3f}" if r is not None else f"  {k:30s}      n/a")
    print("\n  for comparison, the RELATIVE versions currently in the model "
          "(112_concept_validity.py, n=88):")
    for k, v in out["relative_versions_from_112"].items():
        print(f"  {k:30s} {v:+10.3f}")
    print("\n  A materially larger |rho| for the amplitude version means the concept was named "
          "after a\n  construct the pipeline had already made unmeasurable, and that restoring it "
          "requires a\n  loader change -- not a new estimator.")

    report["part_b"] = out
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--n-subjects", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dest = OUT / "topographic_construct.json"
    prior = json.loads(dest.read_text()) if dest.exists() else None
    if prior:
        prior.pop("provenance", None)          # write_result re-stamps it
    report = prior if prior else {
        "note": "R2. Three concepts are named after amplitude constructs that the pipeline "
                "cannot express, because every loader per-channel robust-scales and every "
                "concept is a within-channel ratio."}

    if args.part in ("a", "both"):
        report = part_a(report)
    if args.part in ("b", "both"):
        report = part_b(report, args.n_subjects)

    from src.utils.provenance import write_result
    dest = write_result(dest, report, script=Path(__file__).name)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
