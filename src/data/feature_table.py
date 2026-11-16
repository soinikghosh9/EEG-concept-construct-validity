"""Validated access to a harmonised feature table.

A feature directory holds arrays that must correspond row for row:

    meta.npz              groups / labels / cohort / start, one entry per window
    window_concepts.npy   [n_windows, 15]
    spectral.npy          [n_windows, 123]

Nothing enforced that. The arrays are written by different code paths and one can be regenerated
without the others -- `--skip-spectral` rewrites the meta and leaves the old spectral block in
place -- after which every join by row index silently reads the wrong window. This was live on
disk for several days: `channel_emg15/spectral.npy` held 61,582 rows beside a 65,540-row meta.

No metric reveals that. The model still trains, the AUC still lands in a plausible range, and the
features simply belong to different subjects than the labels. Hence a check that runs before the
data is used rather than a test that runs afterwards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FeatureTableError(RuntimeError):
    """A feature directory is internally inconsistent and must not be used."""


def check_table(directory: Path, require_spectral: bool = False) -> dict:
    """Verify a feature directory is self-consistent. Raises `FeatureTableError` if not.

    Returns a small summary so callers can log what they loaded.
    """
    directory = Path(directory)
    meta_path = directory / "meta.npz"
    if not meta_path.exists():
        raise FeatureTableError(f"{directory} has no meta.npz")

    meta = np.load(meta_path, allow_pickle=False)
    n = len(meta["groups"])
    for key in ("groups", "labels", "cohort"):
        if key not in meta:
            raise FeatureTableError(f"{directory}/meta.npz is missing '{key}'")
        if len(meta[key]) != n:
            raise FeatureTableError(
                f"{directory}/meta.npz: '{key}' has {len(meta[key])} entries against "
                f"{n} windows")

    summary = {"dir": str(directory), "n_windows": int(n),
               "n_subjects": int(len(np.unique(meta["groups"])))}

    for name, expect_cols in (("window_concepts.npy", None), ("spectral.npy", None)):
        p = directory / name
        if not p.exists():
            if name == "spectral.npy" and require_spectral:
                raise FeatureTableError(f"{directory} has no spectral.npy")
            continue
        rows = np.load(p, mmap_mode="r").shape[0]
        if rows != n:
            raise FeatureTableError(
                f"{directory}/{name} has {rows} rows against {n} windows in meta.npz. "
                f"These arrays are joined by row index, so one of them is stale and every "
                f"downstream result would silently use the wrong windows. Re-extract with "
                f"127_extract_harmonised.py.")
        summary[name] = int(rows)
        if expect_cols is not None:
            cols = np.load(p, mmap_mode="r").shape[1]
            if cols != expect_cols:
                raise FeatureTableError(f"{directory}/{name} has {cols} columns, expected "
                                        f"{expect_cols}")

    return summary


def load_table(directory: Path, spectral: bool = False, verbose: bool = True):
    """Load a validated feature table.

    Returns ``(meta, concepts, spectral_or_None)`` after `check_table` has passed.
    """
    summary = check_table(directory, require_spectral=spectral)
    if verbose:
        extra = f", spectral {summary['spectral.npy']}" if "spectral.npy" in summary else ""
        print(f"  feature table {Path(directory).name}: {summary['n_windows']:,} windows, "
              f"{summary['n_subjects']:,} subjects{extra}")
    directory = Path(directory)
    meta = np.load(directory / "meta.npz", allow_pickle=False)
    C = np.load(directory / "window_concepts.npy")
    S = np.load(directory / "spectral.npy") if spectral else None
    return meta, C, S


def check_all(root: Optional[Path] = None) -> int:
    """Validate every feature directory under `root`. Returns the number that failed."""
    root = Path(root or Path(__file__).resolve().parents[2] / "outputs" / "harmonised_features")
    bad = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            s = check_table(d)
            cols = " ".join(f"{k.split('.')[0]}={v}" for k, v in s.items()
                            if k.endswith(".npy"))
            print(f"  OK    {d.name:24s} {s['n_windows']:7,d} windows  "
                  f"{s['n_subjects']:5,d} subjects  {cols}")
        except FeatureTableError as e:
            bad += 1
            print(f"  FAIL  {d.name:24s} {e}")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check_all() else 0)
