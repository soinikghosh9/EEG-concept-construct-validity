"""Shared corpus cache for the 2026-08 audit scripts (117-121).

Every one of those scripts needs the same four things -- per-window subject ids, labels,
augmentation flags and per-subject recording duration -- and each currently pays ~4 minutes to
rebuild the `DatasetFactory` for them. Worse, each rebuild is an opportunity for the configuration
to drift between scripts, which is exactly the class of defect this audit is about.

So: build once, store once, and record the configuration that produced it in the file itself.
Any script that loads this cache is guaranteed to see the same corpus as every other.

The cache is keyed to `enable_augmentation=False`, matching `101_track_a.py` and therefore
matching `outputs/results/track_a/X_full.npy` (73,986 windows) and
`outputs/results/validity/window_concepts_emgfree.npy`. A length mismatch against either is a
hard error rather than a warning, because a silent misalignment between the feature matrix and
the label vector would corrupt every downstream number without any visible symptom.

    python experiments/_audit_cache.py          # build
    from _audit_cache import load_corpus        # use
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

CACHE = ROOT / "outputs" / "results" / "audit" / "corpus_cache.npz"

#: The two cached matrices this corpus must align with, window for window.
X_FULL = ROOT / "outputs" / "results" / "track_a" / "X_full.npy"
W_EMGFREE = ROOT / "outputs" / "results" / "validity" / "window_concepts_emgfree.npy"
W_ORIGINAL = ROOT / "outputs" / "results" / "reliability" / "window_concepts.npy"


def build(force: bool = False) -> Path:
    if CACHE.exists() and not force:
        print(f"cache present: {CACHE}")
        return CACHE

    from _protocol_v2 import (DATASET_PATHS, augmented_mask, cohort_of,  # noqa: E402
                              recording_durations, to_three_class)

    from src.config.config import DATASET_CONFIG
    from src.data.dataset_factory import DatasetFactory

    cfg = dict(DATASET_CONFIG)
    cfg["compute_feature_streams"] = False
    # MUST match 101_track_a.py, which produced X_full.npy. Augmentation off at source, not
    # merely filtered at scoring.
    cfg["enable_augmentation"] = False

    factory = DatasetFactory()
    for name, path in DATASET_PATHS.items():
        if path.exists():
            factory.add_dataset(name, path)
    dataset, groups, labels = factory.create_torch_datasets(config=cfg)
    groups = np.asarray([str(g) for g in groups])
    labels = np.asarray(labels)

    n = len(groups)
    for path in (X_FULL, W_EMGFREE, W_ORIGINAL):
        if path.exists():
            m = np.load(path, mmap_mode="r").shape[0]
            if m != n:
                raise RuntimeError(
                    f"corpus is {n} windows but {path.name} has {m}. These are indexed by the "
                    f"same integers throughout experiments 108-116, so a mismatch means every "
                    f"concept has been attributed to the wrong window. Refusing to cache.")
            print(f"  aligned with {path.name} ({m} windows)")

    dur_map = {str(k): float(v) for k, v in recording_durations(dataset).items()}
    coh = np.array([cohort_of(g) for g in groups])
    is_aug = augmented_mask(dataset)
    y3 = to_three_class(labels)
    dur = np.array([dur_map.get(g, np.nan) for g in groups], dtype=np.float64)

    missing = int(np.isnan(dur).sum())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, groups=groups, labels=labels, y3=y3, cohort=coh, is_aug=is_aug, duration=dur,
        meta=json.dumps({
            "n_windows": int(n),
            "enable_augmentation": False,
            "compute_feature_streams": False,
            "n_windows_missing_duration": missing,
            "aligned_with": [p.name for p in (X_FULL, W_EMGFREE, W_ORIGINAL) if p.exists()],
        }))
    print(f"wrote {CACHE}  ({n:,} windows, {len(np.unique(groups)):,} subjects, "
          f"{missing:,} windows without a duration)")
    return CACHE


def load_corpus():
    """-> dict with groups, labels, y3, cohort, is_aug, duration (all per-window) and meta."""
    if not CACHE.exists():
        build()
    d = np.load(CACHE, allow_pickle=False)
    out = {k: d[k] for k in ("groups", "labels", "y3", "cohort", "is_aug", "duration")}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def subject_table(c, min_windows: int = 8):
    """Per-subject view: ids, 3-class label, cohort, duration, and the window indices.

    `min_windows` matches experiments 108-116 so subject sets are identical across the audit.
    Windows are ordered as they appear in the corpus, so `idx[0::2]` / `idx[1::2]` reproduce the
    interleaved halves used for the split-half statistics.
    """
    groups, y3, coh = c["groups"], c["y3"], c["cohort"]
    real = ~c["is_aug"]
    subs, idxs = [], []
    for s in np.unique(groups):
        i = np.flatnonzero((groups == s) & real)
        if len(i) >= min_windows:
            subs.append(s)
            idxs.append(i)
    subs = np.array(subs)
    return {
        "subjects": subs,
        "index": idxs,
        "y3": np.array([y3[groups == s][0] for s in subs]),
        "cohort": np.array([coh[groups == s][0] for s in subs]),
        "duration": np.array([np.nanmedian(c["duration"][groups == s]) for s in subs]),
    }


if __name__ == "__main__":
    build(force="--force" in sys.argv)
