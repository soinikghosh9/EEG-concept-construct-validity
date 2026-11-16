"""Age: how much of this corpus is separable without any EEG, and does the model ride it?

Patient age is the largest single confound available in a dementia corpus and this project had
never reported it. Within the training cohort, age alone separates the classes at AUC 0.786 for
screening and 0.822 for dementia -- better than several of the deep architectures manage at the
blind site. A paper that reports EEG discrimination without reporting the age baseline is not
reporting the comparison a reader needs.

The reason this is worth a section rather than a footnote is that age behaves exactly like the
duration and epoch-count confounds the manuscript already treats: it is strongly predictive in the
cohort the model is trained on and it REVERSES at the site the model is evaluated on. At ds004504
the controls are older than the patients, so age-only discrimination there falls below chance. Any
model riding age gains in training and loses externally.

Four analyses:

  A. AGE-ONLY DISCRIMINATION, per site and contrast. Establishes the size of the confound in
     training and the sign reversal at evaluation.

  B. INCREMENTAL VALUE. Age alone, concepts alone, and concepts+age, on identical folds. If
     concepts+age is no better than concepts alone, the concept vector already carries the age
     information and adding it explicitly buys nothing.

  C. AGE-MATCHED INTERNAL VALIDATION. Greedy 1:1 nearest-neighbour matching within a caliper, so
     the age distributions of the two arms coincide. What the model retains here is what it knows
     that is not age.

  D. EARLY-ONSET ALZHEIMER'S. The decisive test, and the corpus supplies it as a controlled
     experiment. CAUEEG annotates its Alzheimer's cases as early- or late-onset: eoad (n=39) has a
     mean age of 61.6 against 78.8 for load (n=182) and 65.1 for controls. Early-onset cases are
     therefore YOUNGER than the controls they must be separated from, so an age-based classifier
     scores below chance on exactly the group where a screening instrument is most valuable and a
     clinician is least likely to suspect dementia. If the EEG model detects eoad at all, it is
     doing something age cannot.

    python experiments/173_age_confound_and_earlyonset.py --tag channel_clean
Out: outputs/results/audit/age_confound_<tag>.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result       # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
CAU_ANN = ROOT / "datasets" / "caueeg-dataset" / "annotation.json"
ON_TSV = ROOT / "datasets" / "openneuro_ds004504" / "participants.tsv"
SEED, N_BOOT = 42, 2000


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ok_p, ok_n = np.isfinite(pos), np.isfinite(neg)
    pos, neg = pos[ok_p], neg[ok_n]
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
        return None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    o = [auc(pos[rng.integers(0, len(pos), len(pos))],
             neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    o = [v for v in o if v is not None]
    return a, [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def load_ages():
    """Subject id -> (age, set of annotation symptoms). CAUEEG and the dementia site."""
    age, sym = {}, {}
    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}
    for serial, r in ann.items():
        if r.get("age") is not None:
            age[("CAUEEG", serial)] = float(r["age"])
            sym[("CAUEEG", serial)] = set(r.get("symptom") or [])
    if ON_TSV.exists():
        for r in csv.DictReader(open(ON_TSV, newline="", encoding="utf-8"), delimiter="\t"):
            try:
                age[("ds004504_raw", r["participant_id"].strip())] = float(r["Age"])
            except (TypeError, ValueError, KeyError):
                pass
    return age, sym


def match_age(pos_age, neg_age, caliper=2.0, seed=SEED):
    """Greedy 1:1 nearest-neighbour matching on age within a caliper. Returns index pairs."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pos_age))
    free = list(range(len(neg_age)))
    pairs = []
    for i in order:
        if not free:
            break
        d = [abs(neg_age[j] - pos_age[i]) for j in free]
        k = int(np.argmin(d))
        if d[k] <= caliper:
            pairs.append((int(i), int(free[k])))
            free.pop(k)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--caliper", type=float, default=2.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location(
        "tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    T = tb.subject_table(args.tag)
    subs, coh, cls_ = T["subjects"], T["cohort"], T["cls"]
    X, names = T["X_concept"], T["names"]
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1}
    signs = [DIR[n] for n in names]

    age_map, sym_map = load_ages()

    def key(s, c):
        s = str(s)
        if c == "CAUEEG":
            m = re.search(r"(\d+)$", s)
            return ("CAUEEG", m.group(1).zfill(5)) if m else None
        if c == "ds004504_raw":
            m = re.search(r"(sub-\d+)", s)
            return ("ds004504_raw", m.group(1)) if m else None
        return None

    ages = np.array([age_map.get(key(s, c), np.nan) for s, c in zip(subs, coh)], dtype=float)
    syms = [sym_map.get(key(s, c), set()) for s, c in zip(subs, coh)]
    have = np.isfinite(ages)
    print(f"corpus {args.tag}: {len(subs)} subjects, age available for {int(have.sum())}")
    for c in ("CAUEEG", "ds004504_raw"):
        n = int((have & (coh == c)).sum())
        print(f"   {c:14s} {n:5d} with age")

    report = {"tag": args.tag, "n_subjects": int(len(subs)),
              "n_with_age": int(have.sum()), "caliper_years": args.caliper,
              "note": "Age baseline, incremental value, age-matched validation and the "
                      "early-onset experiment, on the analysed corpus."}

    # ---------------- A. age-only discrimination, per site --------------------------
    print("\n" + "=" * 78)
    print("A. AGE-ONLY DISCRIMINATION")
    print("=" * 78)
    A = {}
    for site in ("CAUEEG", "ds004504_raw"):
        m = have & (coh == site)
        if m.sum() < 20:
            continue
        a_, c_ = ages[m], cls_[m]
        stats = {k: {"n": int((c_ == k).sum()),
                     "age_mean": float(a_[c_ == k].mean()) if (c_ == k).any() else None,
                     "age_sd": float(a_[c_ == k].std(ddof=1)) if (c_ == k).sum() > 1 else None}
                 for k in ("CN", "MCI", "AD", "FTD")}
        site_out = {"age_by_class": stats, "contrasts": {}}
        print(f"\n{site}")
        for k, v in stats.items():
            if v["n"]:
                print(f"   {k:4s} n={v['n']:4d}  age {v['age_mean']:.1f} +/- {v['age_sd']:.1f}")
        for nm, pos_m, neg_m in (
                ("CN_vs_dementia", np.isin(c_, ["AD", "FTD"]), c_ == "CN"),
                ("CN_vs_AD", c_ == "AD", c_ == "CN"),
                ("CN_vs_MCI", c_ == "MCI", c_ == "CN"),
                ("CN_vs_impaired", np.isin(c_, ["AD", "FTD", "MCI"]), c_ == "CN")):
            if pos_m.sum() < 5 or neg_m.sum() < 5:
                continue
            v, ci = auc_ci(a_[pos_m], a_[neg_m])
            site_out["contrasts"][nm] = {"auc_age_only": v, "ci95": ci,
                                         "n_pos": int(pos_m.sum()), "n_neg": int(neg_m.sum())}
            flag = "   <-- BELOW CHANCE" if v is not None and v < 0.5 else ""
            print(f"   {nm:16s} AUC(age) {v:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]{flag}")
        A[site] = site_out
    report["A_age_only"] = A

    # ---------------- helper: fit the concept head, out-of-fold on CAUEEG ------------
    def oof_scores(mask, y, constrained=True, with_age=False):
        """Out-of-fold concept scores for the subjects in `mask`, five-fold, subject-grouped."""
        Xa = X[mask]
        if with_age:
            Xa = np.column_stack([Xa, ages[mask]])
        ya = y[mask]
        out = np.full(len(ya), np.nan)
        sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for tr, te in sk.split(Xa, ya):
            if constrained:
                s = signs + [0] if with_age else signs
                f = SignConstrainedLogistic(s, C=0.5).fit(Xa[tr], ya[tr])
                out[te] = f.predict_proba(Xa[te])[:, 1]
            else:
                sc = StandardScaler().fit(Xa[tr])
                f = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xa[tr]), ya[tr])
                out[te] = f.predict_proba(sc.transform(Xa[te]))[:, 1]
        return out, ya

    # ---------------- B. incremental value of age -----------------------------------
    print("\n" + "=" * 78)
    print("B. INCREMENTAL VALUE OF AGE (within the training cohort, out-of-fold)")
    print("=" * 78)
    B = {}
    for nm, pos, neg in (("CN_vs_dementia", ["AD", "FTD"], ["CN"]),
                         ("CN_vs_MCI", ["MCI"], ["CN"]),
                         ("CN_vs_impaired", ["AD", "FTD", "MCI"], ["CN"])):
        m = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        s_c, yy = oof_scores(m, y, constrained=True, with_age=False)
        s_a, _ = oof_scores(m, y, constrained=True, with_age=True)
        a_age = auc(ages[m][yy == 1], ages[m][yy == 0])
        a_con = auc(s_c[yy == 1], s_c[yy == 0])
        a_both = auc(s_a[yy == 1], s_a[yy == 0])
        B[nm] = {"n": int(m.sum()), "auc_age_only": a_age, "auc_concepts": a_con,
                 "auc_concepts_plus_age": a_both,
                 "delta_from_adding_age": (a_both - a_con) if (a_both and a_con) else None}
        print(f"   {nm:16s} n={int(m.sum()):4d}  age {a_age:.3f} | concepts {a_con:.3f} | "
              f"concepts+age {a_both:.3f}  (delta {a_both - a_con:+.3f})")
    report["B_incremental"] = B

    # ---------------- C. age-matched internal validation ----------------------------
    print("\n" + "=" * 78)
    print(f"C. AGE-MATCHED INTERNAL VALIDATION (caliper {args.caliper} y)")
    print("=" * 78)
    C = {}
    for nm, pos, neg in (("CN_vs_dementia", ["AD", "FTD"], ["CN"]),
                         ("CN_vs_MCI", ["MCI"], ["CN"]),
                         ("CN_vs_impaired", ["AD", "FTD", "MCI"], ["CN"])):
        m = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        s_c, yy = oof_scores(m, y, constrained=True, with_age=False)
        am = ages[m]
        pi = np.flatnonzero(yy == 1)
        ni = np.flatnonzero(yy == 0)
        pairs = match_age(am[pi], am[ni], caliper=args.caliper)
        if len(pairs) < 30:
            continue
        kp = pi[[a for a, _ in pairs]]
        kn = ni[[b for _, b in pairs]]
        smd_before = abs(am[pi].mean() - am[ni].mean()) / np.sqrt(
            (am[pi].var(ddof=1) + am[ni].var(ddof=1)) / 2)
        smd_after = abs(am[kp].mean() - am[kn].mean()) / np.sqrt(
            (am[kp].var(ddof=1) + am[kn].var(ddof=1)) / 2)
        a_un, ci_un = auc_ci(s_c[pi], s_c[ni])
        a_ma, ci_ma = auc_ci(s_c[kp], s_c[kn])
        ag_un = auc(am[pi], am[ni])
        ag_ma = auc(am[kp], am[kn])
        C[nm] = {"n_unmatched": int(m.sum()), "n_pairs": len(pairs),
                 "age_smd_before": float(smd_before), "age_smd_after": float(smd_after),
                 "unmatched": {"concepts": a_un, "ci95": ci_un, "age_only": ag_un},
                 "matched": {"concepts": a_ma, "ci95": ci_ma, "age_only": ag_ma},
                 "concept_auc_lost_to_matching": (a_un - a_ma)}
        print(f"   {nm:16s} pairs={len(pairs):4d}  SMD {smd_before:.2f} -> {smd_after:.2f}")
        print(f"      unmatched  concepts {a_un:.3f}  age {ag_un:.3f}")
        print(f"      matched    concepts {a_ma:.3f} [{ci_ma[0]:.3f}, {ci_ma[1]:.3f}]  "
              f"age {ag_ma:.3f}   (concepts lose {a_un - a_ma:+.3f})")
    report["C_age_matched"] = C

    # ---------------- D. early-onset Alzheimer's ------------------------------------
    print("\n" + "=" * 78)
    print("D. EARLY-ONSET vs LATE-ONSET ALZHEIMER'S")
    print("=" * 78)
    is_cau = have & (coh == "CAUEEG")
    eo = np.array([("eoad" in s) for s in syms]) & is_cau
    lo = np.array([("load" in s) for s in syms]) & is_cau
    cn = (cls_ == "CN") & is_cau
    D = {"n_eoad": int(eo.sum()), "n_load": int(lo.sum()), "n_cn": int(cn.sum()),
         "age_eoad": float(ages[eo].mean()), "age_load": float(ages[lo].mean()),
         "age_cn": float(ages[cn].mean())}
    print(f"   eoad n={D['n_eoad']:3d} age {D['age_eoad']:.1f} | "
          f"load n={D['n_load']:3d} age {D['age_load']:.1f} | "
          f"CN n={D['n_cn']:3d} age {D['age_cn']:.1f}")

    # concept scores for the whole AD-vs-CN problem, out-of-fold, then split by onset
    m = is_cau & (np.isin(cls_, ["AD"]) | cn)
    y = np.isin(cls_, ["AD"]).astype(int)
    s_c, yy = oof_scores(m, y, constrained=True, with_age=False)
    idx = np.flatnonzero(m)
    eo_local = np.array([bool(eo[i]) for i in idx])
    lo_local = np.array([bool(lo[i]) for i in idx])
    cn_local = (yy == 0)
    for nm, sel in (("eoad_vs_CN", eo_local), ("load_vs_CN", lo_local)):
        if sel.sum() < 10:
            continue
        a_m, ci_m = auc_ci(s_c[sel], s_c[cn_local])
        a_a, ci_a = auc_ci(ages[idx][sel], ages[idx][cn_local])
        D[nm] = {"n_pos": int(sel.sum()), "n_neg": int(cn_local.sum()),
                 "auc_concepts": a_m, "ci95_concepts": ci_m,
                 "auc_age_only": a_a, "ci95_age": ci_a}
        flag = "   <-- age BELOW CHANCE" if a_a is not None and a_a < 0.5 else ""
        print(f"   {nm:12s} concepts {a_m:.3f} [{ci_m[0]:.3f}, {ci_m[1]:.3f}]   "
              f"age {a_a:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}]{flag}")
    if "eoad_vs_CN" in D and "load_vs_CN" in D:
        D["concept_gap_load_minus_eoad"] = D["load_vs_CN"]["auc_concepts"] - D["eoad_vs_CN"]["auc_concepts"]
        D["age_gap_load_minus_eoad"] = D["load_vs_CN"]["auc_age_only"] - D["eoad_vs_CN"]["auc_age_only"]
        print(f"\n   load - eoad:  concepts {D['concept_gap_load_minus_eoad']:+.3f}   "
              f"age {D['age_gap_load_minus_eoad']:+.3f}")
    report["D_early_onset"] = D

    # ---------------- E. does the age gain transfer? --------------------------------
    # The decisive test of whether age belongs in the concept set. Age is worth +0.11 AUC
    # inside the training cohort; a concept earns its place only if that survives transfer to a
    # clinic the model has never seen. Both arms are trained on CAUEEG alone and applied
    # unchanged at the dementia site, which is the protocol every other reported result uses.
    print("\n" + "=" * 78)
    print("E. DOES THE AGE GAIN TRANSFER? (train CAUEEG, evaluate at the dementia site)")
    print("=" * 78)
    E = {}
    for nm, pos, neg in (("CN_vs_dementia", ["AD", "FTD"], ["CN"]),
                         ("CN_vs_AD", ["AD"], ["CN"])):
        tr = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        te = have & (coh == "ds004504_raw") & np.isin(cls_, pos + neg)
        if tr.sum() < 50 or te.sum() < 20:
            continue
        y = np.isin(cls_, pos).astype(int)
        row = {"n_train": int(tr.sum()), "n_eval": int(te.sum())}
        for arm, with_age in (("concepts", False), ("concepts_plus_age", True)):
            Xtr = np.column_stack([X[tr], ages[tr]]) if with_age else X[tr]
            Xte = np.column_stack([X[te], ages[te]]) if with_age else X[te]
            # s_j w_j >= 0, so +1 constrains the coefficient upward: older -> more impaired,
            # which is the direction age takes in the training cohort. (-1 here pins it to zero
            # and silently reproduces the concepts-only arm.)
            s = (signs + [+1]) if with_age else signs
            fits = []
            sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
            for a, _ in sk.split(Xtr, y[tr]):
                fits.append(SignConstrainedLogistic(s, C=0.5).fit(Xtr[a], y[tr][a]))
            sc = np.mean([f.predict_proba(Xte)[:, 1] for f in fits], axis=0)
            v, ci = auc_ci(sc[y[te] == 1], sc[y[te] == 0])
            row[arm] = {"auc": v, "ci95": ci}
            print(f"   {nm:16s} {arm:18s} external AUC {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        d_ext = row["concepts_plus_age"]["auc"] - row["concepts"]["auc"]
        d_int = B.get(nm, {}).get("delta_from_adding_age")
        row["delta_external"] = d_ext
        row["delta_internal"] = d_int
        E[nm] = row
        msg = f"   {'':16s} adding age: internal {d_int:+.3f}" if d_int else ""
        print(f"{msg}, external {d_ext:+.3f}")
    report["E_age_transfer"] = E

    dest = write_result(OUT / f"age_confound_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
