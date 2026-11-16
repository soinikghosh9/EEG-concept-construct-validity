"""Sex as a confound and as a subgroup, at the only site that records it.

The age analysis (`173_age_confound_and_earlyonset.py`) asked what a birthdate alone achieves and
how much of the instrument's discrimination survives removing it. Sex is the other demographic that
every clinical model is expected to report against, and this script asks the same questions -- with
one structural difference that has to be stated before any number is read.

THE TRAINING COHORT DOES NOT RECORD SEX. Its annotation carries serial, age and symptom, and
nothing else; the EDF headers are anonymised to sex=0. So three of the five analyses that were
possible for age are impossible for sex: it cannot be added as a concept, the training cohort
cannot be sex-matched, and no internal sex-stratified figure can be produced. What remains is the
external dementia site, which records it for all 87 analysed subjects, and that is where this runs.

  A  How imbalanced is sex at the evaluation site, and what does sex alone achieve there?
  B  Does the instrument's external discrimination survive sex matching?
  C  Does it work equally well in women and in men (the subgroup analysis a clinical model owes)?
  D  Do the named concepts themselves read differently by sex among controls -- that is, is the
     measurement invariant, separately from whether the prediction is fair?

    python experiments/177_sex_confound.py --tag channel_clean
Out: outputs/results/audit/sex_confound_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result       # noqa: E402

OUT = ROOT / "outputs" / "results" / "audit"
PART = ROOT / "datasets" / "openneuro_ds004504_raw" / "participants.tsv"
CAU_ANN = ROOT / "datasets" / "caueeg-dataset" / "annotation.json"
SEED, N_BOOT = 42, 2000

DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return None
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    # mid-ranks, not ordinal ranks: ages are integers and group indicators are binary, and
    # ordinal ranking breaks those ties by array position rather than splitting them.
    r = rankdata(x, method="average")
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(pos, neg, n=N_BOOT, seed=SEED):
    a = auc(pos, neg)
    if a is None:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    o = [auc(pos[rng.integers(0, len(pos), len(pos))],
             neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    o = [v for v in o if v is not None]
    return a, [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    subs, coh, cls_ = T["subjects"], T["cohort"], T["cls"]
    X, names = T["X_concept"], T["names"]
    signs = [DIR[n] for n in names]

    report = {"tag": args.tag,
              "note": "Sex as confound and subgroup. Recorded only at the external dementia site."}

    # =============== 0. establish that the training cohort has no sex field ============
    ann = json.loads(CAU_ANN.read_text())["data"]
    fields = sorted({k for r in ann for k in r})
    has_sex = any(k.lower() in ("sex", "gender") for k in fields)
    report["training_cohort_sex_recorded"] = bool(has_sex)
    report["training_cohort_fields"] = fields
    print(f"training cohort annotation fields: {fields}")
    print(f"   sex recorded in the training cohort: {has_sex}")
    if has_sex:
        print("   [!] a sex field now exists; the analyses below are no longer the only ones possible")

    # =============== A. imbalance and sex alone at the evaluation site =================
    P = pd.read_csv(PART, sep="\t")
    P["pid"] = P.participant_id.str.replace("sub-", "", regex=False).str.lstrip("0")
    sex_by_pid = dict(zip(P.pid, P.Gender))
    grp_by_pid = dict(zip(P.pid, P.Group))

    den = coh == "ds004504_raw"
    def pid(s):
        m = re.search(r"(\d+)", str(s))
        return m.group(1).lstrip("0") if m else None
    sex = np.array([sex_by_pid.get(pid(s), "") if d else "" for s, d in zip(subs, den)])
    matched = int((sex[den] != "").sum())
    print(f"\nevaluation site: {int(den.sum())} subjects in corpus, sex resolved for {matched}")
    assert matched == int(den.sum()), "sex could not be resolved for every evaluation subject"

    print("\n" + "=" * 78)
    print("A. IMBALANCE, AND WHAT SEX ALONE ACHIEVES")
    print("=" * 78)
    A = {}
    female = (sex == "F").astype(float)
    contrasts = {"CN_vs_AD": ("CN", ["AD"]), "CN_vs_dementia": ("CN", ["AD", "FTD"]),
                 "CN_vs_impaired": ("CN", ["AD", "FTD", "MCI"])}
    for nm, (neg_cls, pos_cls) in contrasts.items():
        m = den & np.isin(cls_, [neg_cls] + pos_cls)
        y = np.isin(cls_[m], pos_cls)
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        a, ci = auc_ci(female[m][y], female[m][~y])
        A[nm] = {"n": int(m.sum()), "n_pos": int(y.sum()),
                 "pct_female_pos": float(female[m][y].mean() * 100),
                 "pct_female_neg": float(female[m][~y].mean() * 100),
                 "auc_sex_only": a, "ci95": ci}
        print(f"   {nm:16s} n={int(m.sum()):3d}  female {female[m][y].mean():5.1%} of cases vs "
              f"{female[m][~y].mean():5.1%} of controls   sex-only AUC {a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    report["A_sex_alone"] = A

    # =============== B/C. transferred model, matched and stratified ====================
    print("\n" + "=" * 78)
    print("B. DOES EXTERNAL DISCRIMINATION SURVIVE SEX MATCHING?")
    print("=" * 78)
    B, C = {}, {}
    rng = np.random.default_rng(SEED)
    for nm, (neg_cls, pos_cls) in contrasts.items():
        tr = (coh == "CAUEEG") & np.isin(cls_, [neg_cls] + pos_cls)
        te = den & np.isin(cls_, [neg_cls] + pos_cls)
        ytr = np.isin(cls_[tr], pos_cls).astype(int)
        yte = np.isin(cls_[te], pos_cls)
        if yte.sum() < 5 or (~yte).sum() < 5:
            continue
        fits = [SignConstrainedLogistic(signs, C=0.5).fit(X[tr][a_], ytr[a_])
                for a_, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], ytr)]
        sc = np.mean([f.predict_proba(X[te])[:, 1] for f in fits], axis=0)
        s_te = sex[te]
        a0, ci0 = auc_ci(sc[yte], sc[~yte])

        # sex-matched: take equal numbers of cases and controls within each sex stratum
        keep = np.zeros(len(sc), bool)
        for g in ("F", "M"):
            ip = np.flatnonzero(yte & (s_te == g))
            ineg = np.flatnonzero((~yte) & (s_te == g))
            k = min(len(ip), len(ineg))
            if k:
                keep[rng.choice(ip, k, replace=False)] = True
                keep[rng.choice(ineg, k, replace=False)] = True
        am, cim = auc_ci(sc[keep & yte], sc[keep & ~yte])
        af, _ = auc_ci(female[te][keep & yte], female[te][keep & ~yte])

        # Matching discards cases, so the single draw above could be which ones were kept.
        # Sweep the draw.
        sweep = []
        for sd in range(50):
            r2 = np.random.default_rng(sd)
            k2 = np.zeros(len(sc), bool)
            for g in ("F", "M"):
                ip = np.flatnonzero(yte & (s_te == g))
                ineg = np.flatnonzero((~yte) & (s_te == g))
                k = min(len(ip), len(ineg))
                if k:
                    k2[r2.choice(ip, k, replace=False)] = True
                    k2[r2.choice(ineg, k, replace=False)] = True
            v = auc(sc[k2 & yte], sc[k2 & ~yte])
            if v is not None:
                sweep.append(v)
        sweep = np.array(sweep)
        B[nm] = {"auc_unmatched": a0, "ci95_unmatched": ci0,
                 "n_matched": int(keep.sum()), "auc_matched": am, "ci95_matched": cim,
                 "sex_only_after_matching": af,
                 "auc_lost_to_matching": (a0 - am) if (a0 and am) else None,
                 "matching_seed_sweep": {"n_seeds": len(sweep), "mean": float(sweep.mean()),
                                         "sd": float(sweep.std()),
                                         "range": [float(sweep.min()), float(sweep.max())],
                                         "frac_above_unmatched": float((sweep > a0).mean())}}
        print(f"   {nm:16s} unmatched {a0:.3f} [{ci0[0]:.3f},{ci0[1]:.3f}]  ->  "
              f"sex-matched (n={int(keep.sum())}) {am:.3f} [{cim[0]:.3f},{cim[1]:.3f}]   "
              f"sex-only now {af:.3f}")
        print(f"   {'':16s}   50 matching draws: mean {sweep.mean():.3f} sd {sweep.std():.3f} "
              f"range [{sweep.min():.3f},{sweep.max():.3f}]  exceeds unmatched in "
              f"{(sweep > a0).mean():.0%}")

        # subgroup performance
        for g in ("F", "M"):
            gm = s_te == g
            ag, cig = auc_ci(sc[gm & yte], sc[gm & ~yte])
            if ag is None:
                continue
            C.setdefault(nm, {})[g] = {"n": int(gm.sum()), "n_pos": int((gm & yte).sum()),
                                       "auc": ag, "ci95": cig}
        if nm in C and "F" in C[nm] and "M" in C[nm]:
            C[nm]["difference_F_minus_M"] = C[nm]["F"]["auc"] - C[nm]["M"]["auc"]
            rb = np.random.default_rng(SEED)
            fp = np.flatnonzero((s_te == "F") & yte); fn = np.flatnonzero((s_te == "F") & ~yte)
            mp = np.flatnonzero((s_te == "M") & yte); mn = np.flatnonzero((s_te == "M") & ~yte)
            dd = []
            for _ in range(N_BOOT):
                v1 = auc(sc[rb.choice(fp, len(fp))], sc[rb.choice(fn, len(fn))])
                v2 = auc(sc[rb.choice(mp, len(mp))], sc[rb.choice(mn, len(mn))])
                if v1 is not None and v2 is not None:
                    dd.append(v1 - v2)
            C[nm]["difference_ci95"] = [float(np.percentile(dd, 2.5)),
                                        float(np.percentile(dd, 97.5))]
    report["B_sex_matched"] = B

    print("\n" + "=" * 78)
    print("C. DOES IT WORK EQUALLY WELL IN WOMEN AND MEN?")
    print("=" * 78)
    for nm, d in C.items():
        if "F" in d and "M" in d:
            print(f"   {nm:16s} women n={d['F']['n']:3d} AUC {d['F']['auc']:.3f} "
                  f"[{d['F']['ci95'][0]:.3f},{d['F']['ci95'][1]:.3f}]   "
                  f"men n={d['M']['n']:3d} AUC {d['M']['auc']:.3f} "
                  f"[{d['M']['ci95'][0]:.3f},{d['M']['ci95'][1]:.3f}]   "
                  f"difference {d['difference_F_minus_M']:+.3f} "
                  f"[{d['difference_ci95'][0]:+.3f},{d['difference_ci95'][1]:+.3f}]")
    report["C_subgroup"] = C

    # =============== D. do the concepts read differently by sex? ======================
    print("\n" + "=" * 78)
    print("D. ARE THE CONCEPTS THEMSELVES INVARIANT TO SEX? (controls only)")
    print("=" * 78)
    cn = den & (cls_ == "CN")
    f_, m_ = cn & (sex == "F"), cn & (sex == "M")
    D = {"n_female": int(f_.sum()), "n_male": int(m_.sum()), "concepts": {}}
    print(f"   {int(f_.sum())} female and {int(m_.sum())} male controls")
    from scipy.stats import mannwhitneyu
    ps = []
    for j, nmc in enumerate(names):
        a, b = X[f_, j], X[m_, j]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0
        p = float(mannwhitneyu(a, b).pvalue)
        D["concepts"][nmc] = {"smd_female_minus_male": smd, "p": p}
        ps.append(p)
    # Benjamini-Hochberg
    order = np.argsort(ps)
    q = np.empty(len(ps))
    for rank, j in enumerate(order, 1):
        q[j] = min(1.0, ps[j] * len(ps) / rank)
    for j, nmc in enumerate(names):
        D["concepts"][nmc]["q"] = float(q[j])
    D["n_significant_after_fdr"] = int((q < 0.05).sum())
    for nmc, v in sorted(D["concepts"].items(), key=lambda kv: -abs(kv[1]["smd_female_minus_male"]))[:5]:
        print(f"      {nmc:22s} SMD {v['smd_female_minus_male']:+.3f}  p={v['p']:.3f}  q={v['q']:.3f}")
    print(f"   concepts differing by sex after FDR correction: {D['n_significant_after_fdr']} of {len(names)}")
    report["D_measurement_invariance"] = D

    dest = write_result(OUT / f"sex_confound_{args.tag}.json", report, script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
