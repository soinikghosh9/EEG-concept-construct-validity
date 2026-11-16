"""R1b: does the artefact proxy track independently identified muscle components?

The topographic test (`117_artefact_proxy_validation.py`) did not support the manuscript's claim
that three concepts measure scalp electromyogram: the failed concepts track MIDLINE gamma more
strongly than lateral gamma at every site, and the training cohort shows almost no lateral excess
(1.06x, against 2.34x at OpenNeuro). That test is indirect. This one is as close to a ground truth
as the corpus allows.

ICLabel (Pion-Tonachini et al. 2019) classifies independent components into brain, muscle, eye,
heart, line noise, channel noise and other, from the component's topography, spectrum and
autocorrelation together. It is trained on hand-labelled components and is the standard automated
muscle detector in the field. If `artefact_proxy` -- mean relative 30-45 Hz power -- is a measure
of muscle, it must correlate with the share of a recording's variance carried by components that
ICLabel calls muscle. If it does not, the name is wrong regardless of what else is true.

--------------------------------------------------------------------------------------------
Three design decisions that matter, and why
--------------------------------------------------------------------------------------------

1. **Read from source, never through the project's loaders.** Every loader applies
   `RobustScaler(quantile_range=(25,75))` per channel over the whole recording
   (`caueeg_loader.py:123`, `bids_loader.py:171`). Per-channel rescaling changes the relative
   amplitudes between electrodes, which is exactly what an ICA topography encodes -- and
   ICLabel's strongest muscle cue is a focal, edge-weighted topography. Running ICA on
   loader-scaled data would corrupt the input to the classifier. This script therefore opens the
   original EDF and .set files itself.

2. **Prepare the data the way ICLabel expects.** ICLabel was trained on 1-100 Hz, common-average
   referenced data decomposed by extended infomax. Deviating from any of those degrades it
   silently. Where the sampling rate does not allow 100 Hz the low edge is reduced and the
   subject is flagged, because a muscle classifier deprived of high frequencies is exactly the
   wrong tool to trust quietly.

3. **Both cohorts, reported separately, because they are not comparable.** The OpenNeuro
   derivatives distributed with ds004504 have ALREADY had ICA applied by the dataset authors,
   who removed eye and jaw components; CAUEEG has had none. So OpenNeuro measures RESIDUAL muscle
   after cleaning, and CAUEEG measures muscle as acquired. CAUEEG is the more informative test
   and it is also the cohort the model is trained on.

--------------------------------------------------------------------------------------------
Pre-specified reading
--------------------------------------------------------------------------------------------

The proxy is supported as a muscle measure if, in CAUEEG, the Spearman correlation between
`artefact_proxy` and the muscle variance fraction exceeds 0.4 with a bootstrap interval excluding
zero. Between 0.2 and 0.4, it is a weak and partial indicator and should be renamed. Below 0.2 it
is not a muscle measure and the manuscript's attribution must be withdrawn outright.

The three concepts that failed the audit are tested the same way, along with two that passed
(`pdr_frequency`, `theta_alpha_ratio`) as negative controls: if the controls track muscle just as
well, the analysis is measuring something generic and no verdict can be drawn.

    python experiments/123_ica_muscle_check.py --n-caueeg 60 --n-openneuro 60
Out: outputs/results/validity/ica_muscle_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _audit_cache import load_corpus, subject_table  # noqa: E402
from _protocol_v2 import CN, DEM, MCI, DATASET_PATHS  # noqa: E402

OUT = ROOT / "outputs" / "results" / "validity"
W_ORIGINAL = ROOT / "outputs" / "results" / "reliability" / "window_concepts.npy"
W_EMGFREE = OUT / "window_concepts_emgfree.npy"
SEED = 42

CH19 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
RENAME = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

#: Window-concept column indices (window_concepts.npy has 15 columns; see concepts.py).
WIN_COL = {"pdr_frequency": 0, "theta_alpha_ratio": 7, "aperiodic_exponent": 11,
           "spectral_edge": 12, "spectral_entropy": 13, "artefact_proxy": 14}
FAILED = ["aperiodic_exponent", "spectral_entropy", "spectral_edge"]
CONTROLS = ["pdr_frequency", "theta_alpha_ratio"]

#: Seconds of data used per recording. Extended infomax on 19 channels needs far less than this
#: (~20 x 19^2 = 7,220 samples); the limit is chosen for runtime, and a fixed duration also stops
#: recording length from influencing the decomposition.
SECONDS = 180.0


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


def spearman_ci(a, b, n_boot=5000, seed=SEED):
    r = spearman(a, b)
    if r is None:
        return None, None, None
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    boots = [spearman(a[i], b[i]) for i in rng.integers(0, len(a), (n_boot, len(a)))]
    boots = [x for x in boots if x is not None]
    if len(boots) < 200:
        return r, None, None
    return r, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(v, nd=3):
    return "  n/a " if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def edf_channel_map(ch_names):
    """CAUEEG EDF labels -> canonical 19. Mirrors `caueeg_loader._standardize_channels`.

    The EDF ships channels as ``Fp1-AVG`` and the midline as ``FZ``/``CZ``/``PZ`` with an
    uppercase Z, which no standard montage matches. Reproducing the loader's mapping here rather
    than inventing a second one keeps this script reading the same electrodes the model does.
    """
    fix = {"FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "FP1": "Fp1", "FP2": "Fp2",
           "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    out = {}
    for ch in ch_names:
        base = ch.replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()
        out[ch] = fix.get(base.upper(), base)
    return out


def prepare(raw, verbose=False):
    """Channel-standardise, reference and filter exactly as ICLabel expects.

    Returns (raw, hi) where `hi` is the low-pass edge actually achieved; ICLabel wants 100 Hz and
    a lower value means its high-frequency evidence is truncated.
    """
    import mne

    raw.rename_channels({k: v for k, v in RENAME.items() if k in raw.ch_names})
    keep = [c for c in CH19 if c in raw.ch_names]
    if len(keep) < 19:
        return None, None
    raw.pick(keep)
    raw.reorder_channels(CH19)
    raw.set_channel_types({c: "eeg" for c in CH19})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    nyq = raw.info["sfreq"] / 2.0
    hi = min(100.0, nyq - 5.0)
    if hi <= 30.0:
        return None, None
    raw.filter(1.0, hi, verbose=False)
    # ICLabel is trained on average-referenced data; `projection=False` applies it directly, which
    # is what the classifier expects (a projector left unapplied would not reach the ICA).
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw, hi


def muscle_burden(raw, hi, seed=SEED):
    """-> dict with the share of variance carried by ICLabel-identified muscle components."""
    import mne
    from mne_icalabel import label_components

    n_comp = 18                      # 19 channels minus one lost to the average reference
    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=seed, max_iter="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose=False)
        res = label_components(raw, ica, method="iclabel")

    labels = np.asarray(res["labels"])
    probs = np.asarray(res["y_pred_proba"], dtype=float)

    def share(idx):
        if len(idx) == 0:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = ica.get_explained_variance_ratio(raw, components=list(idx), ch_type="eeg")
        return float(d.get("eeg", 0.0))

    mus = np.flatnonzero(labels == "muscle artifact")
    mus_strict = np.asarray([i for i in mus if probs[i] > 0.8], dtype=int)
    eye = np.flatnonzero(labels == "eye blink")
    brain = np.flatnonzero(labels == "brain")
    return {
        "lowpass_hz": hi,
        "n_muscle": int(len(mus)),
        "n_muscle_p80": int(len(mus_strict)),
        "muscle_var_frac": share(mus),
        "muscle_var_frac_p80": share(mus_strict),
        "eye_var_frac": share(eye),
        "brain_var_frac": share(brain),
        "max_muscle_prob": float(probs[mus].max()) if len(mus) else 0.0,
    }


def caueeg_files(subs):
    """CAUEEG subject id -> source EDF. Ids look like CAUEEG_<CLASS>_<5-digit serial>."""
    root = DATASET_PATHS["CAUEEG"] / "signal" / "edf"
    out = {}
    for s in subs:
        serial = str(s).split("_")[-1]
        p = root / f"{serial}.edf"
        if p.exists():
            out[s] = p
    return out


def openneuro_files(subs):
    root = DATASET_PATHS["ds004504"]
    out = {}
    for s in subs:
        pid = str(s) if str(s).startswith("sub-") else f"sub-{str(s).split('_')[-1]}"
        hits = list((root / "derivatives" / pid).rglob("*.set")) or \
            list(root.rglob(f"{pid}/**/*.set"))
        if hits:
            out[s] = hits[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-caueeg", type=int, default=60)
    ap.add_argument("--n-openneuro", type=int, default=60)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import mne
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(SEED)

    c = load_corpus()
    st = subject_table(c)
    subs, idxs, sub_coh, sub_y3 = (st["subjects"], st["index"], st["cohort"], st["y3"])

    W0 = np.load(W_ORIGINAL, mmap_mode="r")     # pre-repair concepts: the manuscript's claim
    W1 = np.load(W_EMGFREE, mmap_mode="r")      # post-repair

    def conc(src, name):
        col = WIN_COL[name]
        return np.array([float(np.nanmedian(np.asarray(src[ix][:, col], dtype=np.float64)))
                         for ix in idxs])

    pre = {n: conc(W0, n) for n in FAILED + CONTROLS + ["artefact_proxy"]}
    post = {n: conc(W1, n) for n in FAILED}

    report = {
        "note": "R1b. ICLabel muscle-component burden against the artefact proxy and the three "
                "concepts the audit flagged. Source files read directly; the project's loaders "
                "per-channel rescale, which would corrupt ICA topographies.",
        "seconds_per_recording": SECONDS,
        "prespecified": {"support": "rho > 0.4 with CI excluding 0, in CAUEEG",
                         "weak": "0.2 - 0.4, rename", "refute": "< 0.2, withdraw"},
        "cohorts": {},
    }

    plan = []
    if args.n_caueeg:
        plan.append(("CAUEEG", args.n_caueeg, caueeg_files))
    if args.n_openneuro:
        plan.append(("OpenNeuro", args.n_openneuro, openneuro_files))

    for cohort, n_want, resolver in plan:
        m = np.flatnonzero(sub_coh == cohort)
        # stratify by diagnosis so muscle burden is not confounded with class by sampling
        pick = []
        for cls in (CN, MCI, DEM):
            pool = m[sub_y3[m] == cls]
            if len(pool):
                k = min(max(n_want // 3, 1), len(pool))
                pick.extend(rng.choice(pool, k, replace=False).tolist())
        pick = np.array(sorted(set(pick)))
        files = resolver(subs[pick])
        pick = np.array([i for i in pick if subs[i] in files])
        print(f"\n{'='*78}\n{cohort}: {len(pick)} subjects with a source recording")
        if cohort == "OpenNeuro":
            print("  NOTE: these derivatives were ICA-cleaned by the dataset authors "
                  "(eye and jaw\n  components removed), so this measures RESIDUAL muscle.")
        if len(pick) < 20:
            print("  too few; skipping")
            continue

        rows, kept = [], []
        for k, i in enumerate(pick):
            f = files[subs[i]]
            try:
                if f.suffix == ".edf":
                    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                    raw.rename_channels(edf_channel_map(raw.ch_names))
                else:
                    raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
                if raw.times[-1] > SECONDS:
                    raw.crop(0, SECONDS)
                raw, hi = prepare(raw)
                if raw is None:
                    continue
                rows.append(muscle_burden(raw, hi))
                kept.append(i)
            except Exception as e:                                    # noqa: BLE001
                print(f"  [skip] {subs[i]}: {type(e).__name__}: {e}")
                continue
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(pick)} processed", flush=True)

        if len(rows) < 20:
            print(f"  only {len(rows)} usable; skipping {cohort}")
            continue
        kept = np.array(kept)
        mus = np.array([r["muscle_var_frac"] for r in rows])
        mus80 = np.array([r["muscle_var_frac_p80"] for r in rows])
        n_mus = np.array([r["n_muscle"] for r in rows])
        low = np.array([r["lowpass_hz"] for r in rows])

        print(f"\n  {len(rows)} decompositions | low-pass {low.min():.0f}-{low.max():.0f} Hz")
        print(f"  muscle components per recording: median {np.median(n_mus):.0f} "
              f"(range {n_mus.min()}-{n_mus.max()}); "
              f"{100*(n_mus>0).mean():.0f}% of recordings have at least one")
        print(f"  muscle variance fraction: median {np.median(mus):.3f}, "
              f"IQR {np.percentile(mus,25):.3f}-{np.percentile(mus,75):.3f}")

        entry = {"n": len(rows), "lowpass_hz_range": [float(low.min()), float(low.max())],
                 "muscle_components_median": float(np.median(n_mus)),
                 "fraction_with_any_muscle_ic": float((n_mus > 0).mean()),
                 "muscle_var_frac_median": float(np.median(mus)),
                 "correlations": {}}

        print(f"\n  {'quantity':28s} {'rho vs muscle var':>18s} {'95% CI':>20s}")
        targets = [("artefact_proxy (THE PROXY)", pre["artefact_proxy"][kept])]
        targets += [(f"{n} (pre-repair)", pre[n][kept]) for n in FAILED]
        targets += [(f"{n} (repaired)", post[n][kept]) for n in FAILED]
        targets += [(f"{n} (CONTROL)", pre[n][kept]) for n in CONTROLS]
        for label, v in targets:
            r, lo, hi_ = spearman_ci(v, mus)
            entry["correlations"][label] = {"rho": r, "ci95": [lo, hi_]}
            ci = f"[{fmt(lo)},{fmt(hi_)}]" if lo is not None else "n/a"
            print(f"  {label:28s} {fmt(r):>18s} {ci:>20s}")

        # strict-threshold variant, as a robustness check on the p>0.8 subset
        r80, lo80, hi80 = spearman_ci(pre["artefact_proxy"][kept], mus80)
        entry["proxy_vs_muscle_p80"] = {"rho": r80, "ci95": [lo80, hi80]}
        print(f"  {'artefact_proxy vs p>0.8 subset':28s} {fmt(r80):>18s} "
              f"{f'[{fmt(lo80)},{fmt(hi80)}]':>20s}")

        # does muscle burden differ by diagnosis? the confound question, asked directly
        by_class = {}
        for cls, nm in ((CN, "CN"), (MCI, "MCI"), (DEM, "dementia")):
            sel = sub_y3[kept] == cls
            if sel.sum() >= 8:
                by_class[nm] = {"n": int(sel.sum()),
                                "muscle_var_frac_median": float(np.median(mus[sel]))}
        entry["by_class"] = by_class
        print("  muscle burden by diagnosis: " + "  ".join(
            f"{k} n={v['n']} {v['muscle_var_frac_median']:.3f}" for k, v in by_class.items()))

        rho = entry["correlations"]["artefact_proxy (THE PROXY)"]["rho"]
        lo = entry["correlations"]["artefact_proxy (THE PROXY)"]["ci95"][0]
        if rho is None:
            v = "no verdict"
        elif rho > 0.4 and lo is not None and lo > 0:
            v = "SUPPORTED: the proxy tracks ICLabel muscle burden"
        elif rho > 0.2:
            v = "WEAK: partial muscle indicator; the name overstates it"
        else:
            v = "REFUTED: the proxy does not track muscle burden"
        entry["verdict"] = v
        print(f"\n  VERDICT ({cohort}): {v}")
        report["cohorts"][cohort] = entry

    dest = OUT / "ica_muscle_check.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n{'='*78}\nwrote {dest}")
    print("\nCAUEEG carries the verdict: it is the training cohort and it has had no prior ICA. "
          "\nOpenNeuro measures residual muscle after the dataset authors' own cleaning, so a "
          "null\nthere is weaker evidence than a null in CAUEEG.")


if __name__ == "__main__":
    main()
