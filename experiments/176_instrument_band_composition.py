"""What is the artefact instrument made of? Decomposing 30-45 Hz by component class.

The instrument the validity battery uses to convict concepts of tracking artefact is the mean
relative 30-45 Hz power. Every test of it so far has been correlational -- it separates recordings
by labelled myogenic content at AUC 0.764, ocular burden moves it, time on task moves it, and the
lateral topography the electromyographic literature predicts is absent. Those are associations with
the instrument, not a decomposition of it, and they cannot say what fraction of the quantity is
muscle.

This does say. For each recording the signal is decomposed by ICA, the components are classified by
ICLabel, and the signal is then RECONSTRUCTED from each class alone. The 30-45 Hz power of each
reconstruction, as a share of the total across classes, is the composition of the instrument
directly rather than by proxy. `125_artefact_composition.py` asked the same question of BROADBAND
variance; the instrument is a band, and a class can hold little broadband variance while dominating
one band -- which is exactly the case for muscle.

Two further quantities follow from the same decomposition:

  * how much of the instrument survives removing every non-brain component, which is what a
    cleaned pipeline would leave; and
  * whether the instrument, computed on the full signal as the pipeline computes it, tracks the
    muscle-only 30-45 Hz power across recordings -- the association test the earlier work ran,
    but now against a measured muscle quantity rather than a component count.

Recordings are read from source, never through the project's loaders, whose per-channel rescaling
would corrupt the topographies ICLabel classifies on.

    python experiments/176_instrument_band_composition.py --per-cohort 40
Out: outputs/results/audit/instrument_band_composition.json
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _audit_cache import load_corpus, subject_table          # noqa: E402
from _protocol_v2 import CN, DEM, MCI                        # noqa: E402
from src.utils.provenance import write_result                # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
SECONDS = 180.0
SEED = 42

# The instrument's own band definition, from src/data/window_features.py. Reproduced rather than
# imported so that a change there cannot silently redefine what this script measures.
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
CATEGORIES = ["brain", "muscle artifact", "eye blink", "heart beat",
              "line noise", "channel noise", "other"]
SHORT = {"brain": "brain", "muscle artifact": "muscle", "eye blink": "eye",
         "heart beat": "heart", "line noise": "line", "channel noise": "chan",
         "other": "other"}


def _p125():
    spec = importlib.util.spec_from_file_location(
        "p125", ROOT / "experiments" / "125_artefact_composition.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def band_powers(data, sfreq):
    """-> (dict band -> per-channel power, total 0.5-45 Hz per-channel power)."""
    from scipy.signal import welch
    nper = int(min(4.0 * sfreq, data.shape[1]))
    f, P = welch(data, fs=sfreq, nperseg=nper, axis=-1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        out[b] = np.trapezoid(P[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=-1)
    tot = sum(out.values())
    return out, tot


def instrument(data, sfreq):
    """The artefact instrument as the pipeline computes it: mean relative gamma over channels."""
    bp, tot = band_powers(data, sfreq)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(tot > 0, bp["gamma"] / tot, np.nan)
    return float(np.nanmean(rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cohort", type=int, default=40)
    ap.add_argument("--cohorts", default="CAUEEG,ds004504_RAW")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import mne
    from mne_icalabel import label_components
    mne.set_log_level("ERROR")
    p125 = _p125()
    rng = np.random.default_rng(SEED)

    c = load_corpus()
    st = subject_table(c)
    subs, sub_coh, sub_y3 = st["subjects"], st["cohort"], st["y3"]

    report = {"note": "Composition of the 30-45 Hz band by ICLabel component class, by "
                      "reconstruction rather than by association. Read from source.",
              "seconds_per_recording": SECONDS, "bands": {k: list(v) for k, v in BANDS.items()},
              "cohorts": {}}
    want = set(args.cohorts.split(","))

    for label, (corpus_cohort, resolve) in p125.resolvers().items():
        if label not in want:
            continue
        m = np.flatnonzero(sub_coh == corpus_cohort)
        pick = []
        for cls in (CN, MCI, DEM):
            pool = m[sub_y3[m] == cls]
            if len(pool):
                k = min(max(args.per_cohort // 3, 1), len(pool))
                pick.extend(rng.choice(pool, k, replace=False).tolist())
        pick = np.array(sorted(set(pick)))
        print(f"\n{label}: attempting {len(pick)} recordings")

        rows, t0 = [], time.time()
        for n_done, i in enumerate(pick, 1):
            path = resolve(subs[i])
            if path is None:
                continue
            try:
                raw = p125.read_source(Path(path))
                raw.crop(0, min(SECONDS, raw.times[-1]))
                raw, hi, n_ch = p125.prepare(raw)
                if raw is None:
                    continue
                sf = raw.info["sfreq"]
                ica = mne.preprocessing.ICA(n_components=max(n_ch - 1, 5), method="infomax",
                                            fit_params=dict(extended=True), random_state=SEED,
                                            max_iter="auto")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ica.fit(raw, verbose=False)
                    labels = np.asarray(label_components(raw, ica, method="iclabel")["labels"])

                full = raw.get_data()
                bp_full, _ = band_powers(full, sf)
                row = {"subject": str(subs[i]), "cls": int(sub_y3[i]), "n_ch": int(n_ch),
                       "instrument_full": instrument(full, sf),
                       "gamma_abs_full": float(np.mean(bp_full["gamma"])),
                       "gamma_abs": {}, "n_comp": {}}
                for cat in CATEGORIES:
                    idx = list(np.flatnonzero(labels == cat))
                    row["n_comp"][SHORT[cat]] = len(idx)
                    if not idx:
                        row["gamma_abs"][SHORT[cat]] = 0.0
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rec = ica.apply(raw.copy(), include=idx, verbose=False).get_data()
                    bp, _ = band_powers(rec, sf)
                    row["gamma_abs"][SHORT[cat]] = float(np.mean(bp["gamma"]))

                # what a cleaned pipeline would leave: brain components only
                bidx = list(np.flatnonzero(labels == "brain"))
                if bidx:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        brain = ica.apply(raw.copy(), include=bidx, verbose=False).get_data()
                    row["instrument_brain_only"] = instrument(brain, sf)
                rows.append(row)
            except Exception as e:                                   # noqa: BLE001
                print(f"   [skip] {subs[i]}: {type(e).__name__}: {e}")
                continue
            if n_done % 5 == 0:
                el = time.time() - t0
                print(f"   {n_done}/{len(pick)}  {el/n_done:.0f}s each, "
                      f"~{el/n_done*(len(pick)-n_done)/60:.0f} min left")

        if not rows:
            print(f"   [skip] {label}: no usable recordings")
            continue

        # ---- share of the 30-45 Hz band held by each component class ----------------------
        shares = {}
        M = np.array([[r["gamma_abs"][SHORT[c_]] for c_ in CATEGORIES] for r in rows])
        tot = M.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            Sh = np.where(tot > 0, M / tot, np.nan)
        for j, c_ in enumerate(CATEGORIES):
            shares[SHORT[c_]] = {"median": float(np.nanmedian(Sh[:, j])),
                                 "mean": float(np.nanmean(Sh[:, j])),
                                 "iqr": [float(np.nanpercentile(Sh[:, j], 25)),
                                         float(np.nanpercentile(Sh[:, j], 75))],
                                 "n_recordings_with_any": int((M[:, j] > 0).sum())}
        # cross-term check: do the per-class reconstructions account for the full signal's band?
        gfull = np.array([r["gamma_abs_full"] for r in rows])
        ratio = M.sum(axis=1) / np.where(gfull > 0, gfull, np.nan)
        inst = np.array([r["instrument_full"] for r in rows])
        bo = np.array([r.get("instrument_brain_only", np.nan) for r in rows])
        mus = M[:, CATEGORIES.index("muscle artifact")]
        eye = M[:, CATEGORIES.index("eye blink")]

        from scipy.stats import spearmanr
        ent = {"n": len(rows),
               "share_of_30_45Hz_band": shares,
               "sum_of_parts_over_whole": {"median": float(np.nanmedian(ratio)),
                                           "iqr": [float(np.nanpercentile(ratio, 25)),
                                                   float(np.nanpercentile(ratio, 75))],
                                           "note": "1.0 means the class reconstructions account "
                                                   "for the full signal's 30-45 Hz power exactly; "
                                                   "departures are ICA cross-terms"},
               "instrument_full_median": float(np.median(inst)),
               "instrument_brain_only_median": float(np.nanmedian(bo)),
               "instrument_retained_after_cleaning": float(np.nanmedian(bo) / np.median(inst)),
               "spearman_instrument_vs_muscle_gamma": float(spearmanr(inst, mus).statistic),
               "spearman_instrument_vs_eye_gamma": float(spearmanr(inst, eye).statistic),
               "fraction_with_zero_muscle_components": float((M[:, CATEGORIES.index("muscle artifact")] == 0).mean()),
               "rows": rows}
        report["cohorts"][label] = ent

        print(f"\n   {label}  n={len(rows)}")
        print(f"   share of the 30-45 Hz band, median across recordings:")
        for k, v in sorted(shares.items(), key=lambda kv: -kv[1]["median"]):
            print(f"      {k:7s} {v['median']:6.1%}   IQR [{v['iqr'][0]:.1%},{v['iqr'][1]:.1%}]"
                  f"   present in {v['n_recordings_with_any']}/{len(rows)}")
        print(f"   sum of class parts / whole-signal band power: median {np.nanmedian(ratio):.3f}"
              f"  IQR [{np.nanpercentile(ratio,25):.3f},{np.nanpercentile(ratio,75):.3f}]")
        print(f"   instrument, full signal {np.median(inst):.4f} -> brain only "
              f"{np.nanmedian(bo):.4f}  ({ent['instrument_retained_after_cleaning']:.0%} retained)")
        print(f"   instrument vs muscle-only gamma power: rho={ent['spearman_instrument_vs_muscle_gamma']:+.3f}")
        print(f"   instrument vs ocular-only gamma power: rho={ent['spearman_instrument_vs_eye_gamma']:+.3f}")

    dest = write_result(OUT / "instrument_band_composition.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
