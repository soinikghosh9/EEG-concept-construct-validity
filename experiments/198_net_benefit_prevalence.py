"""Clinical utility re-weighted to a screening service's case mix (decision-curve analysis).

Discrimination says how well a score ranks patients; it does not say whether acting on the score
beats what a clinic can already do without it -- refer everyone, or refer nobody. Decision-curve
net benefit answers that. The subtlety this corpus forces is prevalence: CAUEEG's screening
contrast is 59% cases and no triage clinic looks like that, so an unweighted curve overstates the
value of referring. We therefore re-weight cases and controls so the sample behaves like a service
with the stated prevalence (the re-weighting of the increment-over-age literature; see
the standard decision-curve definition), and report net benefit at 5, 10 and 20% prevalence.

The concept score is a product of two stage probabilities and is not itself a calibrated
probability (main text, reliability), so a decision curve on the raw score would be meaningless.
Each fold's score is therefore sigmoid-calibrated on its own training fold and applied out of fold
before the curve is computed. Net benefit is reported for the calibrated concept model against
refer-all and refer-none, and against the age baseline (the only free comparator).

    python experiments/198_net_benefit_prevalence.py --tag channel_clean
Out: outputs/results/audit/net_benefit_prevalence_<tag>.json
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
_FALLBACK = Path(os.environ.get("EEG_DATA_FALLBACK", "datasets"))


def _resolve(rel):
    for base in (ROOT, _FALLBACK):
        if (base / rel).exists():
            return base / rel
    return ROOT / rel


CAU_ANN = _resolve("datasets/caueeg-dataset/annotation.json")

DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}

CONTRASTS = [("CN_vs_impaired", ("AD", "FTD", "MCI"), ("CN",), "Screening"),
             ("CN_vs_dementia", ("AD", "FTD"), ("CN",), "Dementia"),
             ("CN_vs_AD", ("AD",), ("CN",), "Alzheimer's")]
PREVALENCES = [0.05, 0.10, 0.20]


def net_benefit(y, prob, prevalence, thresholds=None):
    """Decision-curve net benefit re-weighted to a target prevalence (after NeuroIPS evaluate.py).

    NB = TP/n_eff - (FP/n_eff)*pt/(1-pt), with cases/controls re-weighted so the sample behaves
    like a population at `prevalence`. Returns per-threshold model / treat-all / treat-none NB.
    """
    y = np.asarray(y); prob = np.asarray(prob, float)
    ok = np.isfinite(prob); y, prob = y[ok], prob[ok]
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.51, 0.01), 3)
    obs = float((y == 1).mean())
    w_pos = prevalence / obs if obs > 0 else 0.0
    w_neg = (1 - prevalence) / (1 - obs) if obs < 1 else 0.0
    n_eff = w_pos * (y == 1).sum() + w_neg * (y == 0).sum()
    rows = []
    for pt in thresholds:
        flag = prob >= pt
        tp = float(((y == 1) & flag).sum()) * w_pos
        fp = float(((y == 0) & flag).sum()) * w_neg
        odds = pt / (1 - pt)
        rows.append({"pt": float(pt),
                     "nb_model": float(tp / n_eff - (fp / n_eff) * odds),
                     "nb_all": float(prevalence - (1 - prevalence) * odds),
                     "nb_none": 0.0})
    return rows


def load_ages():
    age = {}
    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}
    for s, r in ann.items():
        if r.get("age") is not None:
            age[s.zfill(5)] = float(r["age"])
    return age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    subs, coh, cls_ = T["subjects"], T["cohort"], T["cls"]
    X, names = T["X_concept"], T["names"]
    signs = [DIR[n] for n in names]
    age_map = load_ages()
    ser = [re.search(r"(\d+)$", str(s)) for s in subs]
    ages = np.array([age_map.get(m.group(1).zfill(5), np.nan) if (m and c == "CAUEEG") else np.nan
                     for m, c in zip(ser, coh)], float)

    def calibrated_oof(Xa, ya):
        """Per-fold sigmoid-calibrated out-of-fold posteriors for the sign-constrained head."""
        out = np.full(len(ya), np.nan)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya):
            f = SignConstrainedLogistic(signs, C=0.5).fit(Xa[a], ya[a])
            raw_tr = f.predict_proba(Xa[a])[:, 1]
            cal = LogisticRegression(max_iter=1000).fit(_logit(raw_tr).reshape(-1, 1), ya[a])
            raw_te = f.predict_proba(Xa[b])[:, 1]
            out[b] = cal.predict_proba(_logit(raw_te).reshape(-1, 1))[:, 1]
        return out

    def age_oof(am, ya):
        out = np.full(len(ya), np.nan)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(am.reshape(-1, 1), ya):
            f = LogisticRegression(max_iter=1000).fit(am[a].reshape(-1, 1), ya[a])
            out[b] = f.predict_proba(am[b].reshape(-1, 1))[:, 1]
        return out

    report = {"tag": args.tag, "seed": SEED, "prevalences": PREVALENCES,
              "note": "DCA re-weighted to service prevalence; concept posteriors per-fold "
                      "sigmoid-calibrated out of fold.", "contrasts": {}}

    for name, pos, neg, label in CONTRASTS:
        m = (coh == "CAUEEG") & np.isin(cls_, pos + neg) & np.isfinite(ages)
        y = np.isin(cls_, pos).astype(int)[m]
        p_con = calibrated_oof(X[m], y)
        p_age = age_oof(ages[m], y)
        obs_prev = float(y.mean())
        entry = {"label": label, "n": int(m.sum()), "n_pos": int(y.sum()),
                 "observed_prevalence": obs_prev, "by_prevalence": {}}
        print(f"\n[{label}] n={int(m.sum())} pos={int(y.sum())} obs_prev={obs_prev:.2f}")
        for prev in PREVALENCES:
            cm = net_benefit(y, p_con, prev)
            am = net_benefit(y, p_age, prev)
            # range of pt where the model beats both defaults, and its peak gain over refer-all
            wins = [r["pt"] for r in cm if r["nb_model"] > max(r["nb_all"], 0) + 1e-9]
            gains_all = [(r["nb_model"] - max(r["nb_all"], 0.0)) for r in cm]
            gains_age = [c["nb_model"] - a["nb_model"] for c, a in zip(cm, am)]
            peak_all = max(gains_all)
            pt_peak = cm[int(np.argmax(gains_all))]["pt"]
            peak_age = max(gains_age)
            pt_peak_age = cm[int(np.argmax(gains_age))]["pt"]
            entry["by_prevalence"][f"{prev:.2f}"] = {
                "win_range_pt": [min(wins), max(wins)] if wins else None,
                "peak_extra_per_1000_vs_refer_all": round(peak_all * 1000, 1),
                "pt_at_peak_vs_all": pt_peak,
                "peak_extra_per_1000_vs_age": round(peak_age * 1000, 1),
                "pt_at_peak_vs_age": pt_peak_age,
                "curve_model": [round(r["nb_model"], 5) for r in cm],
                "curve_all": [round(r["nb_all"], 5) for r in cm],
                "curve_age": [round(r["nb_model"], 5) for r in am],
                "thresholds": [r["pt"] for r in cm]}
            wr = entry["by_prevalence"][f"{prev:.2f}"]["win_range_pt"]
            print(f"   prev {prev:.0%}: beats defaults over pt {wr};  "
                  f"peak +{peak_all*1000:.0f}/1000 vs refer-all (pt {pt_peak}); "
                  f"+{peak_age*1000:.0f}/1000 vs age (pt {pt_peak_age})")
        report["contrasts"][name] = entry

    dest = write_result(OUT / f"net_benefit_prevalence_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    main()
