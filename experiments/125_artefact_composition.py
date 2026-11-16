"""How much of each cohort is not brain, and would cleaning it change anything?

This is the evidence base for the rebuild decision. The proposal is to add filtering, notch,
re-referencing and ICA to a pipeline that currently applies none of them (see
`AUDIT_RUN_RESULTS_2026-08-04.md` §P). That is roughly two days of compute, and it is only worth
spending if the artefact burden it would remove is actually large. A single pilot recording
suggested 39.9 per cent of CAUEEG's variance sits in ocular components, but n=1 is not a basis for
a two-day job.

So: decompose a stratified sample of every readable cohort, label the components with ICLabel, and
report the share of variance in each category. Three questions, each of which changes the plan:

  Q1  **How much non-brain variance is there?** If the training cohort is 60 per cent brain and
      40 per cent artefact, ICA is the single largest lever available and the rebuild is
      justified on this number alone. If it is 90 per cent brain, the deep-model case is weak and
      I should say so.

  Q2  **Does artefact burden differ BY DIAGNOSIS?** This is the confound question asked directly,
      at the level of physiology rather than of a spectral proxy. A class difference in ocular or
      muscle variance is a route to the label that contains no disease information, and it would
      be present in every model trained on this data.

  Q3  **What did ds004504's authors' cleaning actually do?** The raw and derivative versions of
      the same 88 recordings are now both on disk. Decomposing both measures the effect of the
      ASR and ICA they applied -- which is the cross-cohort asymmetry the manuscript does not
      currently disclose, quantified rather than asserted.

Recordings are read from source and NEVER through the project's loaders, whose per-channel
`RobustScaler` would corrupt the topographies ICLabel classifies on.

    python experiments/125_artefact_composition.py --per-cohort 40
Out: outputs/results/audit/artefact_composition.json
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _audit_cache import load_corpus, subject_table  # noqa: E402
from _protocol_v2 import CN, DEM, MCI, DATASET_PATHS  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
RAW_ON = ROOT / "datasets" / "openneuro_ds004504_raw"
SECONDS = 180.0
SEED = 42

CH19 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
CATEGORIES = ["brain", "muscle artifact", "eye blink", "heart beat",
              "line noise", "channel noise", "other"]
SHORT = {"brain": "brain", "muscle artifact": "muscle", "eye blink": "eye",
         "heart beat": "heart", "line noise": "line", "channel noise": "chan",
         "other": "other"}


def edf_channel_map(ch_names):
    fix = {"FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "FP1": "Fp1", "FP2": "Fp2",
           "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    out = {}
    for ch in ch_names:
        base = ch.replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()
        out[ch] = fix.get(base.upper(), base)
    return out


def prepare(raw, drop_flat=True):
    """Standardise, drop dead channels, filter 1-100 Hz, average reference."""
    import mne

    # Canonicalise case-insensitively. GENEEG labels its channels FP1/FZ/CZ/PZ in upper case, and
    # a case-sensitive match against CH19 silently dropped Fp1, Fp2, Fz, Cz and Pz -- leaving
    # exactly 12 of 17 electrodes. Because Fp1 and Fp2 are where blinks are largest, that made
    # GENEEG's ocular variance read as 0.000 and its brain share as 0.971. A montage bug can
    # manufacture a clean-looking cohort, which is worth stating plainly.
    canon = {c.upper(): c for c in CH19}
    canon.update({"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"})
    raw.rename_channels({ch: canon[ch.upper()] for ch in raw.ch_names if ch.upper() in canon})
    keep = [c for c in CH19 if c in raw.ch_names]
    if len(keep) < 12:
        return None, None, None
    raw.pick(keep)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})

    if drop_flat:
        # GENEEG supplies 17 of 19 channels and the loader zero-fills T5/T6. A constant channel
        # makes the data rank-deficient, which silently degrades ICA; dropping it is correct and
        # is recorded so the reader knows the decomposition used fewer sensors.
        d = raw.get_data()
        flat = [ch for ch, v in zip(raw.ch_names, d.std(axis=1)) if v < 1e-12]
        if flat:
            raw.drop_channels(flat)
    if len(raw.ch_names) < 12:
        return None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    nyq = raw.info["sfreq"] / 2.0
    hi = min(100.0, nyq - 5.0)
    if hi <= 30.0:
        return None, None, None
    raw.filter(1.0, hi, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw, hi, len(raw.ch_names)


def decompose(raw, n_ch, seed=SEED):
    """-> {category: variance share}. Shares are over the categories, so they sum to ~1."""
    import mne
    from mne_icalabel import label_components

    ica = mne.preprocessing.ICA(n_components=max(n_ch - 1, 5), method="infomax",
                                fit_params=dict(extended=True), random_state=seed,
                                max_iter="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose=False)
        res = label_components(raw, ica, method="iclabel")
    labels = np.asarray(res["labels"])

    out = {}
    for cat in CATEGORIES:
        idx = list(np.flatnonzero(labels == cat))
        if not idx:
            out[SHORT[cat]] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = ica.get_explained_variance_ratio(raw, components=idx, ch_type="eeg")
        out[SHORT[cat]] = float(d.get("eeg", 0.0))
    return out


def read_source(path):
    import mne
    if path.suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.rename_channels(edf_channel_map(raw.ch_names))
        return raw
    if path.suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    if path.suffix == ".eeg":                      # GENEEG ASCII, [T, 17]
        from src.data.geneeg_loader import _GENEEG_CH
        arr = np.loadtxt(path, dtype=np.float64)
        info = mne.create_info(list(_GENEEG_CH), sfreq=250.0, ch_types="eeg")
        return mne.io.RawArray(arr.T * 1e-6, info, verbose="ERROR")
    raise ValueError(f"unhandled suffix {path.suffix}")


def resolvers():
    """cohort label -> (subject-id -> source path) resolver."""
    def caueeg(sid):
        p = DATASET_PATHS["CAUEEG"] / "signal" / "edf" / f"{str(sid).split('_')[-1]}.edf"
        return p if p.exists() else None

    def on_raw(sid):
        pid = str(sid) if str(sid).startswith("sub-") else f"sub-{str(sid).split('_')[-1]}"
        hits = list((RAW_ON / pid / "eeg").glob("*.set"))
        return hits[0] if hits else None

    def on_deriv(sid):
        pid = str(sid) if str(sid).startswith("sub-") else f"sub-{str(sid).split('_')[-1]}"
        hits = list((DATASET_PATHS["ds004504"] / "derivatives" / pid).rglob("*.set"))
        return hits[0] if hits else None

    def geneeg(sid):
        stem = str(sid).split("_", 2)[-1]
        hits = list(DATASET_PATHS["GENEEG"].rglob(f"{stem}.eeg"))
        return hits[0] if hits else None

    return {"CAUEEG": ("CAUEEG", caueeg),
            "ds004504_RAW": ("OpenNeuro", on_raw),
            "ds004504_derivative": ("OpenNeuro", on_deriv),
            "GENEEG": ("GENEEG", geneeg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cohort", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import mne
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(SEED)

    c = load_corpus()
    st = subject_table(c)
    subs, sub_coh, sub_y3 = st["subjects"], st["cohort"], st["y3"]

    report = {"note": "Variance share by ICLabel component category, per cohort and per class. "
                      "Read from source; the project's loaders per-channel rescale, which would "
                      "corrupt ICA topographies.",
              "seconds_per_recording": SECONDS, "cohorts": {}}

    for label, (corpus_cohort, resolve) in resolvers().items():
        m = np.flatnonzero(sub_coh == corpus_cohort)
        pick = []
        for cls in (CN, MCI, DEM):
            pool = m[sub_y3[m] == cls]
            if len(pool):
                k = min(max(args.per_cohort // 3, 1), len(pool))
                pick.extend(rng.choice(pool, k, replace=False).tolist())
        pick = np.array(sorted(set(pick)))

        rows, classes, nchs = [], [], []
        print(f"\n{'='*76}\n{label}: attempting {len(pick)} subjects", flush=True)
        for k, i in enumerate(pick):
            p = resolve(subs[i])
            if p is None:
                continue
            try:
                raw = read_source(p)
                if raw.times[-1] > SECONDS:
                    raw.crop(0, SECONDS)
                raw, hi, n_ch = prepare(raw)
                if raw is None:
                    continue
                rows.append(decompose(raw, n_ch))
                classes.append(int(sub_y3[i]))
                nchs.append(n_ch)
            except Exception as e:                                   # noqa: BLE001
                print(f"  [skip] {subs[i]}: {type(e).__name__}: {str(e)[:70]}")
                continue
            if (k + 1) % 15 == 0:
                print(f"  {k+1}/{len(pick)}", flush=True)

        if len(rows) < 10:
            print(f"  only {len(rows)} usable; skipped")
            continue
        classes = np.array(classes)
        cats = [SHORT[c_] for c_ in CATEGORIES]
        M = np.array([[r[c_] for c_ in cats] for r in rows])

        print(f"\n  {len(rows)} decompositions, {int(np.median(nchs))} channels (median)")
        print(f"  {'category':10s} {'median share':>13s} {'IQR':>18s}")
        med = {}
        for j, c_ in enumerate(cats):
            q1, q3 = np.percentile(M[:, j], [25, 75])
            med[c_] = float(np.median(M[:, j]))
            print(f"  {c_:10s} {np.median(M[:, j]):13.3f} {f'{q1:.3f}-{q3:.3f}':>18s}")
        nonbrain = float(np.median(M[:, 1:].sum(axis=1)))
        print(f"  {'NON-BRAIN':10s} {nonbrain:13.3f}  <- the headroom ICA cleaning would address")

        by_class = {}
        for cls, nm in ((CN, "CN"), (MCI, "MCI"), (DEM, "dementia")):
            sel = classes == cls
            if sel.sum() >= 8:
                by_class[nm] = {"n": int(sel.sum()),
                                **{c_: float(np.median(M[sel, j])) for j, c_ in enumerate(cats)}}
        if len(by_class) >= 2:
            print(f"\n  by class ({', '.join(f'{k} n={v[chr(110)]}' for k, v in by_class.items())}):")
            print(f"  {'category':10s} " + " ".join(f"{k:>9s}" for k in by_class))
            for c_ in cats:
                vals = [by_class[k][c_] for k in by_class]
                flag = "  <-- differs" if max(vals) - min(vals) > 0.05 else ""
                print(f"  {c_:10s} " + " ".join(f"{v:9.3f}" for v in vals) + flag)

        report["cohorts"][label] = {
            "n": len(rows), "median_channels": int(np.median(nchs)),
            "median_share": med, "nonbrain_median": nonbrain, "by_class": by_class}

    # Q3: what the ds004504 authors' cleaning removed
    a = report["cohorts"].get("ds004504_RAW")
    b = report["cohorts"].get("ds004504_derivative")
    if a and b:
        print(f"\n{'='*76}\nQ3. Effect of the ds004504 authors' ASR + ICA, same 88 recordings:")
        print(f"  {'category':10s} {'raw':>8s} {'derivative':>11s} {'change':>9s}")
        delta = {}
        for c_ in [SHORT[x] for x in CATEGORIES]:
            d = b["median_share"][c_] - a["median_share"][c_]
            delta[c_] = d
            print(f"  {c_:10s} {a['median_share'][c_]:8.3f} {b['median_share'][c_]:11.3f} "
                  f"{d:+9.3f}")
        report["ds004504_cleaning_effect"] = delta

    dest = write_result(OUT / "artefact_composition.json", report,
                        script="125_artefact_composition.py")
    print(f"\nwrote {dest}")
    print("\nHOW TO READ THIS. `nonbrain_median` in CAUEEG is the headroom the rebuild's ICA step "
          "would\naddress in the training cohort. A class difference in eye or muscle share is a "
          "route to the\nlabel containing no disease information, and would be present in every "
          "model trained here.")


if __name__ == "__main__":
    main()
