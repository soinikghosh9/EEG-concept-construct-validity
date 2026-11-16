"""Increment over age: what the recording adds to the free demographic baseline.

Age costs nothing (every clinic holds the date of birth) and is the strongest single predictor
in a dementia corpus, so the reference an EEG instrument must be read against is age, not chance.
A single AUC runs together two questions a clinic faces very differently:

    substitution margin   Delta_sub = AUC(concepts)      - AUC(age)
    incremental margin     Delta_inc = AUC(concepts+age)  - AUC(age)

A clinic never chooses between the recording and the calendar (substitution); it always asks
whether buying the recording improves on what it already holds (increment). Both are reported
here on identical subjects, internally (CAUEEG out-of-fold) and at transfer (train CAUEEG,
evaluate at the site that contributed nothing to training). The increment is tested with a
permutation null that refits the model inside every permutation with the recording replaced by
noise -- the honest test that adding the recording to age beats adding noise to age -- and both
margins carry a paired subject bootstrap. This is the evaluation contract of the increment-over-
age literature (Marquand 2016; Rutherford 2022; Engemann 2022), applied to the concept model.

Age enters the concepts+age arm UNCONSTRAINED (sign 0): age is a baseline covariate, not a concept
with a clinical construct direction under test, so the fair EEG+age arm lets the model use it freely.

    python experiments/196_increment_over_age.py --tag channel_clean --n_perm 200
Out: outputs/results/audit/increment_over_age_<tag>.json
"""
from __future__ import annotations

import argparse
import csv
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
SEED = 42

# Data roots: prefer the project tree, fall back to the detached parent corpus (paths.py contract).
_FALLBACK = Path(os.environ.get("EEG_DATA_FALLBACK", "datasets"))


def _resolve(rel: str) -> Path:
    for base in (ROOT, _FALLBACK):
        p = base / rel
        if p.exists():
            return p
    return ROOT / rel


CAU_ANN = _resolve("datasets/caueeg-dataset/annotation.json")
ON_TSV = _resolve("datasets/openneuro_ds004504_raw/participants.tsv")

# Sign directions for the fourteen classifier concepts (from 157/173; 0 = unconstrained).
DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}

# (positive classes, negative classes, external cohort, label)
CONTRASTS = [
    ("CN_vs_dementia", ("AD", "FTD"), ("CN",), "ds004504_raw", "Dementia"),
    ("CN_vs_AD",       ("AD",),       ("CN",), "ds004504_raw", "Alzheimer's"),
    ("CN_vs_impaired", ("AD", "FTD", "MCI"), ("CN",), "ds004504_raw", "Screening"),
    ("CN_vs_MCI",      ("MCI",),      ("CN",), "GENEEG", "Prodromal"),
]


def auc(pos, neg):
    from scipy.stats import rankdata
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return None
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    r = rankdata(x, method="average")
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_from_scores(y, s):
    return auc(s[y == 1], s[y == 0])


def load_ages():
    age = {}
    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}
    for serial, r in ann.items():
        if r.get("age") is not None:
            age[("CAUEEG", serial)] = float(r["age"])
    if ON_TSV.exists():
        for r in csv.DictReader(open(ON_TSV, newline="", encoding="utf-8"), delimiter="\t"):
            try:
                age[("ds004504_raw", r["participant_id"].strip())] = float(r["Age"])
            except (TypeError, ValueError, KeyError):
                pass
    return age


def key(s, c):
    s = str(s)
    if c == "CAUEEG":
        m = re.search(r"(\d+)$", s)
        return ("CAUEEG", m.group(1).zfill(5)) if m else None
    if c == "ds004504_raw":
        m = re.search(r"(sub-\d+)", s)
        return ("ds004504_raw", m.group(1)) if m else None
    return None


def paired_bootstrap_margin(y, s_eeg, s_both, age, n=2000, seed=SEED):
    """Paired subject bootstrap for Delta_sub and Delta_inc on a fixed set of scores.

    Resamples subjects with replacement; recomputes AUC(age), AUC(eeg), AUC(eeg+age) on each
    resample and both margins from them. Returns observed value, 95% CI and a two-sided p.
    """
    y = np.asarray(y); s_eeg = np.asarray(s_eeg, float)
    s_both = np.asarray(s_both, float); age = np.asarray(age, float)
    a_age = _auc_from_scores(y, age); a_eeg = _auc_from_scores(y, s_eeg)
    a_both = _auc_from_scores(y, s_both)
    obs_sub = a_eeg - a_age
    obs_inc = a_both - a_age
    rng = np.random.default_rng(seed)
    subd, incd = [], []
    n_sub = len(y)
    for _ in range(n):
        i = rng.integers(0, n_sub, n_sub)
        if len(np.unique(y[i])) < 2:
            continue
        aa = _auc_from_scores(y[i], age[i])
        ee = _auc_from_scores(y[i], s_eeg[i])
        bb = _auc_from_scores(y[i], s_both[i])
        if None in (aa, ee, bb):
            continue
        subd.append(ee - aa); incd.append(bb - aa)
    def summ(obs, d):
        d = np.asarray(d, float)
        if d.size == 0:
            return {"delta": float(obs), "ci": [None, None], "p": None}
        p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
        return {"delta": float(obs), "ci": [float(np.percentile(d, 2.5)),
                                            float(np.percentile(d, 97.5))],
                "p": float(min(p, 1.0))}
    return {"auc_age": a_age, "auc_eeg": a_eeg, "auc_eeg_age": a_both,
            "substitution": summ(obs_sub, subd), "increment": summ(obs_inc, incd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    subs, coh, cls_ = T["subjects"], T["cohort"], T["cls"]
    X, names = T["X_concept"], T["names"]
    signs = [DIR[n] for n in names]
    age_map = load_ages()
    ages = np.array([age_map.get(key(s, c), np.nan) for s, c in zip(subs, coh)], float)
    have = np.isfinite(ages)

    print(f"corpus {args.tag}: {len(subs)} subjects, age for {int(have.sum())}")
    print(f"CAUEEG ann: {CAU_ANN}\nds004504 tsv: {ON_TSV}\n")

    rng = np.random.default_rng(SEED)
    report = {"tag": args.tag, "n_perm": args.n_perm, "seed": SEED,
              "age_sign": "unconstrained (0)", "contrasts": {}}

    def fit_oof(mask, y, with_age):
        Xa = np.column_stack([X[mask], ages[mask]]) if with_age else X[mask]
        s = signs + [0] if with_age else signs
        ya = y[mask]
        out = np.full(len(ya), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya):
            f = SignConstrainedLogistic(s, C=0.5).fit(Xa[tr], ya[tr])
            out[te] = f.predict_proba(Xa[te])[:, 1]
        return out, ya

    def fit_transfer(trm, tem, y, with_age):
        Xtr = np.column_stack([X[trm], ages[trm]]) if with_age else X[trm]
        Xte = np.column_stack([X[tem], ages[tem]]) if with_age else X[tem]
        s = signs + [0] if with_age else signs
        fits = [SignConstrainedLogistic(s, C=0.5).fit(Xtr[a], y[trm][a])
                for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y[trm])]
        return np.mean([f.predict_proba(Xte)[:, 1] for f in fits], axis=0)

    for name, pos, neg, ext, label in CONTRASTS:
        entry = {"label": label, "external_cohort": ext}

        # -------- internal (CAUEEG out-of-fold) --------
        m = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        s_c, yy = fit_oof(m, y, False)
        s_ca, _ = fit_oof(m, y, True)
        am = ages[m]
        internal = paired_bootstrap_margin(yy, s_c, s_ca, am)
        internal["n"] = int(m.sum()); internal["n_pos"] = int(yy.sum())

        # permutation-refit increment null: recording (concepts) replaced by noise, age kept with label
        obs_inc = internal["increment"]["delta"]
        perm_ge = 1
        for _ in range(args.n_perm):
            Xperm = X.copy()
            idx = np.flatnonzero(m)
            Xperm[idx] = X[idx][rng.permutation(len(idx))]
            Xa = np.column_stack([Xperm[m], am]); sfit = signs + [0]
            outp = np.full(len(yy), np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, yy):
                f = SignConstrainedLogistic(sfit, C=0.5).fit(Xa[tr], yy[tr])
                outp[te] = f.predict_proba(Xa[te])[:, 1]
            inc_p = _auc_from_scores(yy, outp) - internal["auc_age"]
            if inc_p >= obs_inc - 1e-12:
                perm_ge += 1
        internal["increment"]["p_permutation_refit"] = perm_ge / (args.n_perm + 1)
        entry["internal"] = internal

        # -------- transfer (train CAUEEG, evaluate external) --------
        tem = have & (coh == ext) & np.isin(cls_, pos + neg)
        trm = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        if tem.sum() >= 20 and len(np.unique(y[tem])) > 1 and np.isfinite(ages[tem]).all():
            s_c_e = fit_transfer(trm, tem, y, False)
            s_ca_e = fit_transfer(trm, tem, y, True)
            ext_res = paired_bootstrap_margin(y[tem], s_c_e, s_ca_e, ages[tem])
            ext_res["n"] = int(tem.sum()); ext_res["n_pos"] = int(y[tem].sum())

            obs_inc_e = ext_res["increment"]["delta"]
            perm_ge = 1
            for _ in range(args.n_perm):
                Xperm = X.copy()
                idx = np.flatnonzero(trm)
                Xperm[idx] = X[idx][rng.permutation(len(idx))]
                Xtr = np.column_stack([Xperm[trm], ages[trm]])
                Xte = np.column_stack([X[tem], ages[tem]]); sfit = signs + [0]
                fits = [SignConstrainedLogistic(sfit, C=0.5).fit(Xtr[a], y[trm][a])
                        for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y[trm])]
                sc = np.mean([f.predict_proba(Xte)[:, 1] for f in fits], axis=0)
                inc_p = _auc_from_scores(y[tem], sc) - ext_res["auc_age"]
                if inc_p >= obs_inc_e - 1e-12:
                    perm_ge += 1
            ext_res["increment"]["p_permutation_refit"] = perm_ge / (args.n_perm + 1)
            entry["external"] = ext_res
        else:
            entry["external"] = {"skipped": f"no age at {ext}" if tem.sum() else "no external subjects"}

        report["contrasts"][name] = entry

        def show(tag, r):
            if "skipped" in r:
                print(f"  {tag:9s} {r['skipped']}"); return
            print(f"  {tag:9s} n={r['n']:4d} pos={r['n_pos']:3d} | age {r['auc_age']:.3f}  "
                  f"eeg {r['auc_eeg']:.3f}  eeg+age {r['auc_eeg_age']:.3f}")
            su, ic = r["substitution"], r["increment"]
            print(f"            Delta_sub {su['delta']:+.3f} [{su['ci'][0]:+.3f},{su['ci'][1]:+.3f}] "
                  f"p={su['p']:.3f}   Delta_inc {ic['delta']:+.3f} "
                  f"[{ic['ci'][0]:+.3f},{ic['ci'][1]:+.3f}] p_boot={ic['p']:.3f} "
                  f"p_perm={ic.get('p_permutation_refit'):.3f}")
        print(f"\n[{label}] ({name})")
        show("internal", entry["internal"]); show("external", entry["external"])

    dest = write_result(OUT / f"increment_over_age_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
