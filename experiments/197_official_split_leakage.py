"""Patient-disjoint leakage check on CAUEEG's official no-overlap benchmark split.

The reported benchmark trains under five-fold subject-grouped cross-validation. CAUEEG ships a
per-recording ``serial`` but no patient identifier, so subject-grouping groups recordings, not
patients: if one patient contributed more than one recording they could fall on both sides of a
fold and inflate the internal figure. The dataset authors, who hold the patient identifiers,
publish an official patient-disjoint split (``dementia-no-overlap.json`` and
``abnormal-no-overlap.json``) in which no patient appears in more than one partition. Relabelling
that split with our contrasts and training on its train serials, then evaluating on its held-out
test serials, is the only patient-level check the corpus allows.

For each contrast this reports, on identical subjects (the union of the official partitions):
  * five-fold recording-stratified CV out-of-fold AUC (our standard internal figure), and
  * the official patient-disjoint held-out AUC (train on train+validation serials, test on test),
for both the reported concept model and the reference spectral forest. If the held-out figure
covers the CV figure, subject-grouping is not hiding patient leakage.

    python experiments/197_official_split_leakage.py --tag channel_clean
Out: outputs/results/audit/official_split_leakage_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result  # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
SEED, N_BOOT = 42, 2000
_FALLBACK = Path(os.environ.get("EEG_DATA_FALLBACK", "datasets"))


def _resolve(rel: str) -> Path:
    for base in (ROOT, _FALLBACK):
        p = base / rel
        if p.exists():
            return p
    return ROOT / rel


SPLIT_DEM = _resolve("datasets/caueeg-dataset/dementia-no-overlap.json")
SPLIT_ABN = _resolve("datasets/caueeg-dataset/abnormal-no-overlap.json")

DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}

# (positive classes, negative classes, split file, label)
CONTRASTS = [
    ("CN_vs_impaired", ("AD", "FTD", "MCI"), ("CN",), "abnormal", "Screening"),
    ("CN_vs_dementia", ("AD", "FTD"),        ("CN",), "dementia", "Dementia"),
    ("CN_vs_AD",       ("AD",),              ("CN",), "dementia", "Alzheimer's"),
    ("CN_vs_MCI",      ("MCI",),             ("CN",), "dementia", "Prodromal"),
]


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(s, method="average")
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, s, n=N_BOOT, seed=SEED):
    a = auc(y, s)
    if a is None:
        return a, [None, None]
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s, float)
    o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v = auc(y[i], s[i])
        if v is not None:
            o.append(v)
    return a, [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def load_partition(kind):
    """serial (5-digit) -> 'train' | 'val' | 'test' for the chosen official split."""
    path = SPLIT_DEM if kind == "dementia" else SPLIT_ABN
    d = json.loads(path.read_text())
    part = {}
    for split_key, tag in (("train_split", "train"), ("validation_split", "val"),
                           ("test_split", "test")):
        for r in d.get(split_key, []):
            part[str(r["serial"]).zfill(5)] = tag
    return part, path


def serial_of(s):
    m = re.search(r"(\d+)$", str(s))
    return m.group(1).zfill(5) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    subs, coh, cls_ = T["subjects"], T["cohort"], T["cls"]
    Xc, Xs, names = T["X_concept"], T["X_spectral"], T["names"]
    signs = [DIR[n] for n in names]
    serials = np.array([serial_of(s) if c == "CAUEEG" else None
                        for s, c in zip(subs, coh)], dtype=object)

    print(f"corpus {args.tag}: {len(subs)} subjects; CAUEEG {int((coh=='CAUEEG').sum())}")
    print(f"dementia split: {SPLIT_DEM}\nabnormal split: {SPLIT_ABN}\n")

    def concept_cv(Xtr, ytr, Xte):
        fits = [SignConstrainedLogistic(signs, C=0.5).fit(Xtr[a], ytr[a])
                for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr)]
        return np.mean([f.predict_proba(Xte)[:, 1] for f in fits], axis=0)

    def forest_cv(Xtr, ytr, Xte):
        fits = [RandomForestClassifier(400, random_state=SEED, n_jobs=-1,
                                       class_weight="balanced").fit(Xtr[a], ytr[a])
                for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr)]
        return np.mean([f.predict_proba(Xte)[:, 1] for f in fits], axis=0)

    def oof(kind, X, y):
        out = np.full(len(y), np.nan)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            if kind == "concept":
                f = SignConstrainedLogistic(signs, C=0.5).fit(X[a], y[a])
            else:
                f = RandomForestClassifier(400, random_state=SEED, n_jobs=-1,
                                           class_weight="balanced").fit(X[a], y[a])
            out[b] = f.predict_proba(X[b])[:, 1]
        return out

    report = {"tag": args.tag, "n_boot": N_BOOT, "seed": SEED, "contrasts": {}}

    for name, pos, neg, kind, label in CONTRASTS:
        part, _ = load_partition(kind)
        in_split = np.array([(c == "CAUEEG" and part.get(sr) is not None)
                             for sr, c in zip(serials, coh)])
        m = in_split & np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        tag_arr = np.array([part.get(sr) if c == "CAUEEG" else None
                            for sr, c in zip(serials, coh)], dtype=object)
        tr = m & np.isin(tag_arr, ["train", "val"])
        te = m & (tag_arr == "test")
        if tr.sum() < 40 or te.sum() < 15 or len(np.unique(y[te])) < 2:
            report["contrasts"][name] = {"label": label, "skipped":
                                         f"insufficient (train {int(tr.sum())}, test {int(te.sum())})"}
            print(f"[{label}] skipped (train {int(tr.sum())}, test {int(te.sum())})")
            continue

        entry = {"label": label, "split": kind, "n_union": int(m.sum()),
                 "n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "n_test_pos": int(y[te].sum()), "models": {}}

        for mk, oofk in (("concept_sign_constrained", "concept"), ("spectral_forest", "forest")):
            X = Xc if oofk == "concept" else Xs
            # CV on the union (identical subjects), our standard internal figure
            cv = oof(oofk, X[m], y[m])
            a_cv, ci_cv = boot_ci(y[m], cv)
            # official patient-disjoint held-out
            pred = concept_cv(X[tr], y[tr], X[te]) if oofk == "concept" else forest_cv(X[tr], y[tr], X[te])
            a_ho, ci_ho = boot_ci(y[te], pred)
            entry["models"][mk] = {
                "cv_oof_auc": a_cv, "cv_oof_ci": ci_cv,
                "patient_disjoint_auc": a_ho, "patient_disjoint_ci": ci_ho,
                "delta_disjoint_minus_cv": (a_ho - a_cv) if (a_ho and a_cv) else None}
            print(f"[{label:11s}] {mk:24s} CV {a_cv:.3f} [{ci_cv[0]:.3f},{ci_cv[1]:.3f}]  "
                  f"patient-disjoint {a_ho:.3f} [{ci_ho[0]:.3f},{ci_ho[1]:.3f}]  "
                  f"d={a_ho-a_cv:+.3f}")
        report["contrasts"][name] = entry
        print()

    dest = write_result(OUT / f"official_split_leakage_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
