"""Blinded reader study: inter-reader agreement, reader-estimator agreement, and clinical reading.

Two clinicians graded the 60 blinded recordings rendered by 146_export_reader_traces.py, seeing
age and (where recorded) sex only. This script opens the sealed key and computes the three
quantities the protocol pre-specified (supplementary section "Blinded reader study: protocol"):

  1. agreement between the two readers, which bounds what any estimator could achieve;
  2. agreement between each reader and the estimator of the same name (ICC for frequency,
     rank or kappa statistics for the graded items);
  3. the rate at which a reader records no posterior rhythm where the estimator returned a number.

It adds, as secondary and clearly labelled analyses: whether the battery's artefact instrument
tracks the artefact the readers grade, how the readers' global impression separates the sealed
diagnoses on these 60 against the concept model's score on the same subjects, and whether the
two readers' sheets are internally coherent.

    python experiments/266_reader_study_analysis.py
Out: outputs/results/validity/reader_study_results.json
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

SHEETS = {"R1": ROOT / "reader_study" / "reader1_grades.csv",
          "R2": ROOT / "reader_study" / "reader2_grades.csv"}
KEY = ROOT / "outputs" / "results" / "validity" / "reader_manifest_SEALED.csv"
FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "validity" / "reader_study_results.json"
SEED, NB = 20261116, 5000

COLS = ["id", "pdr", "wf", "pmax", "slow", "pat", "asym", "art", "artloc", "drow", "glob",
        "ungr", "notes"]
ORD = {"slow": {"none": 0, "mild": 1, "moderate": 2, "severe": 3},
       "art": {"none": 0, "mild": 1, "marked": 2},
       "drow": {"none": 0, "possible": 1, "definite": 2},
       "glob": {"normal": 0, "borderline": 1, "abnormal": 2},
       "pmax": {"no": 0, "yes": 1}, "asym": {"no": 0, "possible": 1, "yes": 2},
       "ungr": {"no": 0, "yes": 1}}
# window_concepts.npy column order (15 columns; pdr_variability is computed at aggregation)
WCOL = ["pdr_frequency", "pdr_prominence", "posterior_maximality", "ap_gradient_alpha",
        "ap_gradient_freq", "asymmetry_alpha", "asymmetry_freq", "theta_alpha_ratio",
        "slowing_ratio", "rel_alpha", "rel_theta", "aperiodic_exponent", "spectral_edge",
        "spectral_entropy", "artefact_proxy"]


# ----------------------------------------------------------------------------- reading the sheets
def read_sheet(path):
    d = pd.read_csv(path)
    d.columns = COLS
    d["id"] = d["id"].astype(str).str.strip()
    out = pd.DataFrame({"id": d["id"]})
    p = d["pdr"].astype(str).str.strip().str.lower()
    out["pdr_absent"] = (p == "absent").astype(int)
    out["pdr"] = pd.to_numeric(p.where(p != "absent"), errors="coerce")
    out["wf"] = pd.to_numeric(d["wf"], errors="coerce")
    for k in ORD:
        out[k] = d[k].astype(str).str.strip().str.lower().map(ORD[k])
    out["pat"] = d["pat"].astype(str).str.strip().str.lower()
    out["artloc"] = d["artloc"].astype(str).str.strip().str.lower()
    out["notes"] = d["notes"].fillna("").astype(str)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- statistics
def icc_a1(x, y):
    """ICC(2,1), two-way random, absolute agreement, single measurement (McGraw & Wong A-1)."""
    X = np.column_stack([x, y]).astype(float)
    X = X[np.isfinite(X).all(1)]
    n, k = X.shape
    if n < 5:
        return np.nan
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def kappa(a, b, weights=None, cats=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m].astype(int), b[m].astype(int)
    # kappa is undefined, not zero, when either rater uses a single category: chance agreement
    # then equals observed agreement whatever happens. Report percent agreement instead.
    if len(set(a)) < 2 or len(set(b)) < 2:
        return np.nan
    cats = sorted(set(a) | set(b)) if cats is None else cats
    K = len(cats)
    if K < 2:
        return np.nan
    ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((K, K))
    for u, v in zip(a, b):
        O[ix[u], ix[v]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((K, K))
    W = {None: (i != j).astype(float), "linear": np.abs(i - j) / (K - 1),
         "quadratic": ((i - j) / (K - 1)) ** 2}[weights]
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r = spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


def boot(fn, n, strat=None, nb=NB, seed=SEED):
    """Percentile CI by resampling recordings (stratified by class where given)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        if strat is None:
            ix = rng.integers(0, n, n)
        else:
            ix = np.concatenate([rng.choice(np.flatnonzero(strat == g), (strat == g).sum())
                                 for g in np.unique(strat)])
        v = fn(ix)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < nb * 0.5:
        return [np.nan, np.nan]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def r3(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


# ----------------------------------------------------------------------------- estimator values
def estimator_table(subjects):
    """Subject-level concepts for the 60 reader recordings, NaN preserved (not imputed)."""
    from src.models.v6 import aggregate_subject
    d = FEAT / "channel_clean"
    W = np.load(d / "window_concepts.npy")
    g = np.load(d / "meta.npz")["groups"]
    rows = {}
    for s in subjects:
        i = np.flatnonzero(g == s)
        agg = aggregate_subject(W[i])          # 17-d, CONCEPT_NAMES order, NaN preserved
        w = W[i]
        rows[s] = {"pdr_frequency": agg[0], "pdr_prominence": agg[1], "pdr_variability": agg[2],
                   **{n: np.nanmedian(w[:, WCOL.index(n)]) for n in WCOL[2:]},
                   "pdr_missing_frac": float(np.mean(~np.isfinite(w[:, 0]))),
                   "n_windows": len(i)}
    # amplitude-referenced forms (global scaling arm, same subjects and windows)
    dg = FEAT / "global_clean"
    A = np.load(dg / "window_amp_concepts.npy")
    gg = np.load(dg / "meta.npz")["groups"]
    for s in subjects:
        i = np.flatnonzero(gg == s)
        with np.errstate(all="ignore"):
            med = np.nanmedian(A[i], axis=0) if len(i) else np.full(3, np.nan)
        rows[s].update({"amp_posterior_maximality": med[0], "amp_ap_gradient_alpha": med[1],
                        "amp_asymmetry_alpha": med[2]})
    return pd.DataFrame.from_dict(rows, orient="index")


def model_scores(subjects):
    """Concept-model screening score (CN vs any impairment) for each reader recording.

    CAUEEG recordings get their out-of-fold score under the paper's five folds; ds004504
    recordings get the fold ensemble applied unchanged, exactly as in the external evaluation.
    """
    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic
    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1}
    T = tb.subject_table("channel_clean")
    pos, neg, trc, evc = tb.TASKS["CN_vs_impaired"]
    keep = np.isin(T["cls"], pos + neg)
    y = np.isin(T["cls"], pos).astype(int)
    tr = keep & (T["cohort"] == trc)
    X = T["X_concept"]
    dirs = [DIR[n] for n in T["names"]]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=tb.SEED).split(X[tr], y[tr]))
    tr_idx = np.flatnonzero(tr)
    score = np.full(len(X), np.nan)
    fits = []
    for a, b in folds:
        f = SignConstrainedLogistic(dirs, C=0.5).fit(X[tr_idx[a]], y[tr_idx[a]])
        fits.append(f)
        score[tr_idx[b]] = f.predict_proba(X[tr_idx[b]])[:, 1]
    ext = T["cohort"] == evc
    score[ext] = np.mean([f.predict_proba(X[ext])[:, 1] for f in fits], axis=0)
    sub = {s: i for i, s in enumerate(T["subjects"])}
    return {s: float(score[sub[s]]) for s in subjects}


# ----------------------------------------------------------------------------- main
def main():
    key = pd.DataFrame(list(csv.DictReader(KEY.open(encoding="utf-8"))))
    R = {k: read_sheet(p) for k, p in SHEETS.items()}
    for k in R:
        assert list(R[k]["id"]) == list(key["blind_id"]), f"{k}: row order differs from the key"
    n = len(key)
    est = estimator_table(list(key["group_id"]))
    est = est.loc[key["group_id"]].reset_index(drop=True)
    # The relative estimator is the mean rank of the occipital channels in alpha power, 1 being
    # strongest, so a SMALLER value means more posterior-maximal. Orient it so that larger means
    # more maximal, like the amplitude form (a log posterior/anterior ratio), before any sign is
    # read against a reader's yes/no.
    est["posterior_maximality"] = -est["posterior_maximality"]
    msc = model_scores(list(key["group_id"]))
    key["model_score"] = [msc[s] for s in key["group_id"]]
    key["age"] = pd.to_numeric(key["age"], errors="coerce")
    key["impaired"] = (key["stratum"] != "CN").astype(int)
    strat = key["stratum"].values
    res = {"n_recordings": n,
           "strata": key.groupby(["stratum", "cohort"]).size().rename("n").reset_index()
           .to_dict("records"),
           "readers": {"R1": "reader1_grades.csv", "R2": "reader2_grades.csv"}}
    A, B = R["R1"], R["R2"]

    # ---------------------------------------------------------------- 0. completeness, coherence
    coh = {}
    for k, d in R.items():
        incoh = d[(d["pdr_absent"] == 1) & (d["slow"] >= 2) & (d["glob"] == 0)]
        incoh2 = d[(d["slow"] >= 2) & (d["glob"] == 0)]
        coh[k] = {"cells_blank": int(d[["pmax", "slow", "art", "drow", "glob", "ungr"]]
                                     .isna().sum().sum()),
                  "pdr_absent": int(d["pdr_absent"].sum()),
                  "ungradable": int(d["ungr"].sum()),
                  "global_normal_with_moderate_slowing": len(incoh2),
                  "global_normal_with_absent_pdr_and_moderate_slowing": incoh["id"].tolist(),
                  "notes_filled": int((d["notes"].str.len() > 0).sum()),
                  "distinct_note_openings": int(d.loc[d["notes"].str.len() > 0, "notes"]
                                                .str.split().str[:4].str.join(" ").nunique()),
                  "global": d["glob"].value_counts().sort_index().to_dict(),
                  "slowing": d["slow"].value_counts().sort_index().to_dict(),
                  "artefact": d["art"].value_counts().sort_index().to_dict(),
                  "asymmetry": d["asym"].value_counts().sort_index().to_dict(),
                  "drowsiness": d["drow"].value_counts().sort_index().to_dict()}
    res["sheet_coherence"] = coh

    # ---------------------------------------------------------------- 1. inter-reader agreement
    ir = {}
    both = A["pdr"].notna() & B["pdr"].notna()
    diff = (A["pdr"] - B["pdr"])[both]
    ir["pdr_frequency"] = {
        "n_both_numeric": int(both.sum()),
        "icc_a1": r3(icc_a1(A["pdr"], B["pdr"])),
        "icc_ci": [r3(v) for v in boot(lambda ix: icc_a1(A["pdr"].values[ix], B["pdr"].values[ix]), n)],
        "mean_diff_R1_minus_R2": r3(diff.mean()), "sd_diff": r3(diff.std(ddof=1)),
        "loa": [r3(diff.mean() - 1.96 * diff.std(ddof=1)), r3(diff.mean() + 1.96 * diff.std(ddof=1))],
        "within_1hz": r3(float((diff.abs() <= 1.0).mean())),
        "spearman": r3(rho(A["pdr"], B["pdr"])[0])}
    for item, w in [("pdr_absent", None), ("wf", "quadratic"), ("pmax", None),
                    ("slow", "quadratic"), ("art", "quadratic"), ("drow", "quadratic"),
                    ("glob", "quadratic"), ("asym", None), ("ungr", None)]:
        a, b = A[item].values.astype(float), B[item].values.astype(float)
        m = np.isfinite(a) & np.isfinite(b)
        ir[item] = {"weights": w or "unweighted", "n": int(m.sum()),
                    "kappa": r3(kappa(a, b, w)),
                    "kappa_ci": [r3(v) for v in boot(lambda ix: kappa(a[ix], b[ix], w), n)],
                    "percent_agreement": r3(float((a[m] == b[m]).mean())),
                    "prevalence_R1": r3(float(np.mean(a[m] > 0))),
                    "prevalence_R2": r3(float(np.mean(b[m] > 0)))}
    # slowing present (any) as a binary call, the clinically decisive one
    sa, sb = (A["slow"] > 0).astype(int), (B["slow"] > 0).astype(int)
    ir["slowing_present"] = {"kappa": r3(kappa(sa, sb)), "percent_agreement": r3(float((sa == sb).mean())),
                             "kappa_ci": [r3(v) for v in boot(lambda ix: kappa(sa.values[ix], sb.values[ix]), n)]}
    ga, gb = (A["glob"] > 0).astype(int), (B["glob"] > 0).astype(int)
    ir["global_not_normal"] = {"kappa": r3(kappa(ga, gb)), "percent_agreement": r3(float((ga == gb).mean())),
                               "kappa_ci": [r3(v) for v in boot(lambda ix: kappa(ga.values[ix], gb.values[ix]), n)]}
    res["inter_reader"] = ir

    # ---------------------------------------------------------------- 2. reader vs estimator
    re_ = {}
    mean_pdr = (A["pdr"] + B["pdr"]) / 2
    for lab, r in [("R1", A["pdr"]), ("R2", B["pdr"]), ("mean_of_readers", mean_pdr)]:
        x = est["pdr_frequency"]
        m = r.notna() & x.notna()
        d = (x - r)[m]
        re_[f"pdr_frequency_vs_{lab}"] = {
            "n": int(m.sum()), "icc_a1": r3(icc_a1(x, r)),
            "icc_ci": [r3(v) for v in boot(lambda ix: icc_a1(x.values[ix], r.values[ix]), n)],
            "spearman": r3(rho(x, r)[0]), "p": r3(rho(x, r)[1]),
            "bias_estimator_minus_reader_hz": r3(d.mean()), "sd_diff": r3(d.std(ddof=1)),
            "within_1hz": r3(float((d.abs() <= 1.0).mean()))}
    # absent-rhythm calls the estimator cannot make
    for lab, d in [("R1", A), ("R2", B)]:
        ab = d["pdr_absent"] == 1
        re_[f"pdr_absent_{lab}"] = {
            "n_absent": int(ab.sum()),
            "estimator_returned_number": int((ab & est["pdr_frequency"].notna()).sum()),
            "estimator_value_hz_when_absent": [r3(v) for v in est.loc[ab, "pdr_frequency"]],
            "estimator_value_hz_when_present_median": r3(est.loc[~ab, "pdr_frequency"].median()),
            "missing_window_frac_auc_absent": r3(auc(ab.astype(int), est["pdr_missing_frac"])),
            "prominence_auc_absent_vs_present": r3(auc(ab.astype(int), -est["pdr_prominence"])),
            "estimator_freq_auc_absent_vs_present": r3(auc(ab.astype(int), -est["pdr_frequency"]))}
    both_absent = (A["pdr_absent"] == 1) & (B["pdr_absent"] == 1)
    re_["pdr_absent_both"] = {"n": int(both_absent.sum()),
                              "estimator_returned_number": int((both_absent & est["pdr_frequency"].notna()).sum())}
    pairs = [("wf", "pdr_prominence"), ("pmax", "posterior_maximality"),
             ("pmax", "amp_posterior_maximality"),
             ("slow", "rel_theta"), ("slow", "theta_alpha_ratio"), ("slow", "slowing_ratio"),
             ("slow", "rel_alpha"), ("slow", "aperiodic_exponent"), ("slow", "pdr_frequency"),
             ("art", "artefact_proxy"), ("drow", "rel_theta"), ("drow", "theta_alpha_ratio"),
             ("asym", "asymmetry_alpha"), ("asym", "amp_asymmetry_alpha")]
    for item, conc in pairs:
        for lab, d in [("R1", A), ("R2", B)]:
            r_, p_, nn = rho(d[item], est[conc])
            re_[f"{item}~{conc}|{lab}"] = {"spearman": r3(r_), "p": r3(p_), "n": nn}
    # consensus slowing (mean grade) and artefact
    for item, conc in [("slow", "rel_theta"), ("slow", "theta_alpha_ratio"),
                       ("slow", "slowing_ratio"), ("art", "artefact_proxy"),
                       ("wf", "pdr_prominence"), ("pmax", "posterior_maximality"),
                       ("pmax", "amp_posterior_maximality"), ("drow", "theta_alpha_ratio")]:
        cons = (A[item] + B[item]) / 2
        r_, p_, nn = rho(cons, est[conc])
        ci = boot(lambda ix: rho(cons.values[ix], est[conc].values[ix])[0], n)
        re_[f"{item}~{conc}|consensus"] = {"spearman": r3(r_), "p": r3(p_), "n": nn,
                                           "ci": [r3(v) for v in ci]}
    # artefact by location: muscle graded frontal/temporal/diffuse vs instrument
    for lab, d in [("R1", A), ("R2", B)]:
        re_[f"artefact_marked_auc_instrument|{lab}"] = r3(auc((d["art"] == 2).astype(int),
                                                                est["artefact_proxy"]))
    res["reader_vs_estimator"] = re_
    res["estimator_summary"] = {c: {"median": r3(est[c].median()),
                                    "iqr": [r3(est[c].quantile(.25)), r3(est[c].quantile(.75))],
                                    "n_missing": int(est[c].isna().sum())}
                                for c in ["pdr_frequency", "pdr_prominence", "posterior_maximality",
                                          "amp_posterior_maximality", "asymmetry_alpha",
                                          "amp_asymmetry_alpha", "rel_theta", "artefact_proxy"]}

    # ---------------------------------------------------------------- 3. clinical reading vs key
    cl = {}
    contrasts = {"impaired_vs_CN": (["MCI", "dementia"], ["CN"]),
                 "dementia_vs_CN": (["dementia"], ["CN"]),
                 "MCI_vs_CN": (["MCI"], ["CN"]),
                 "dementia_vs_MCI": (["dementia"], ["MCI"])}
    scores = {"R1_global": A["glob"], "R2_global": B["glob"],
              "R1_slowing": A["slow"], "R2_slowing": B["slow"],
              "readers_mean_global": (A["glob"] + B["glob"]) / 2,
              "readers_mean_slowing": (A["slow"] + B["slow"]) / 2,
              "R1_pdr_neg": -A["pdr"].fillna(4.0), "R2_pdr_neg": -B["pdr"].fillna(4.0),
              "concept_model": key["model_score"], "age": key["age"]}
    for cname, (pos, neg) in contrasts.items():
        m = key["stratum"].isin(pos + neg).values
        y = key["stratum"].isin(pos).values[m].astype(int)
        cl[cname] = {"n": int(m.sum()), "n_pos": int(y.sum())}
        for sname, s in scores.items():
            sv = np.asarray(s, float)[m]
            ci = boot(lambda ix: auc(y[ix], sv[ix]), len(y), strat=y)
            cl[cname][sname] = {"auc": r3(auc(y, sv)), "ci": [r3(v) for v in ci]}
        # paired difference: concept model minus readers' mean global impression
        sm = np.asarray(key["model_score"], float)[m]
        sr = np.asarray(scores["readers_mean_global"], float)[m]
        d0 = auc(y, sm) - auc(y, sr)
        ci = boot(lambda ix: auc(y[ix], sm[ix]) - auc(y[ix], sr[ix]), len(y), strat=y)
        cl[cname]["model_minus_readers_global"] = {"diff": r3(d0), "ci": [r3(v) for v in ci]}
    # Within-cohort contrasts. Pooled over the 60, every MCI case is CAUEEG while half the
    # controls are ds004504, and the model score is out-of-fold at one site and a transferred
    # ensemble at the other, so a pooled AUC partly ranks cohorts. These are the primary ones.
    wc = {}
    for cohort, cname, pos, neg in [("ds004504_raw", "dementia_vs_CN", ["dementia"], ["CN"]),
                                    ("CAUEEG", "dementia_vs_CN", ["dementia"], ["CN"]),
                                    ("CAUEEG", "MCI_vs_CN", ["MCI"], ["CN"]),
                                    ("CAUEEG", "impaired_vs_CN", ["MCI", "dementia"], ["CN"])]:
        m = ((key["cohort"] == cohort) & key["stratum"].isin(pos + neg)).values
        y = key["stratum"].isin(pos).values[m].astype(int)
        row = {"n": int(m.sum()), "n_pos": int(y.sum())}
        for sname in ["readers_mean_global", "readers_mean_slowing", "R1_pdr_neg", "R2_pdr_neg",
                      "concept_model", "age"]:
            sv = np.asarray(scores[sname], float)[m]
            ci = boot(lambda ix: auc(y[ix], sv[ix]), len(y), strat=y)
            row[sname] = {"auc": r3(auc(y, sv)), "ci": [r3(v) for v in ci]}
        sm = np.asarray(key["model_score"], float)[m]
        sr = np.asarray(scores["readers_mean_global"], float)[m]
        ci = boot(lambda ix: auc(y[ix], sm[ix]) - auc(y[ix], sr[ix]), len(y), strat=y)
        row["model_minus_readers_global"] = {"diff": r3(auc(y, sm) - auc(y, sr)),
                                             "ci": [r3(v) for v in ci]}
        wc[f"{cohort}|{cname}"] = row
    cl["within_cohort"] = wc
    # binary reading as a referral rule: any reader-called abnormality
    for lab, d in [("R1", A), ("R2", B)]:
        call = (d["glob"] > 0).astype(int).values
        y = key["impaired"].values
        cl[f"{lab}_not_normal_as_referral"] = {
            "sensitivity": r3(call[y == 1].mean()), "specificity": r3(1 - call[y == 0].mean()),
            "by_stratum_not_normal": {s: r3(call[key["stratum"].values == s].mean())
                                      for s in ["CN", "MCI", "dementia"]}}
    # within-cohort: does the cohort shift the reading in controls?
    cn = key["stratum"] == "CN"
    for lab, d in [("R1", A), ("R2", B)]:
        cl[f"{lab}_controls_by_cohort"] = {
            c: {"slowing_any": r3((d.loc[cn & (key["cohort"] == c), "slow"] > 0).mean()),
                "pdr_median": r3(d.loc[cn & (key["cohort"] == c), "pdr"].median()),
                "artefact_marked": r3((d.loc[cn & (key["cohort"] == c), "art"] == 2).mean())}
            for c in ["CAUEEG", "ds004504_raw"]}
    # reading vs model agreement
    for lab, d in [("R1", A), ("R2", B)]:
        r_, p_, nn = rho(d["glob"], key["model_score"])
        cl[f"{lab}_global~model_score"] = {"spearman": r3(r_), "p": r3(p_)}
    res["clinical_reading"] = cl

    # ---------------------------------------------------------------- 4. sensitivity: gradable only
    g = (A["ungr"] == 0) & (B["ungr"] == 0)
    res["gradable_only"] = {
        "n": int(g.sum()),
        "pdr_icc_readers": r3(icc_a1(A["pdr"][g], B["pdr"][g])),
        "slow_kappa_quadratic": r3(kappa(A["slow"][g], B["slow"][g], "quadratic")),
        "glob_kappa_quadratic": r3(kappa(A["glob"][g], B["glob"][g], "quadratic")),
        "pdr_est_vs_mean_icc": r3(icc_a1(est["pdr_frequency"][g], mean_pdr[g]))}
    # Sensitivity: drop the Reader 1 rows whose global impression contradicts the same reader's
    # own items (absent rhythm and moderate continuous slowing graded globally normal).
    bad = A["id"].isin(coh["R1"]["global_normal_with_absent_pdr_and_moderate_slowing"])
    k = ~bad
    res["without_incoherent_R1_rows"] = {
        "n": int(k.sum()),
        "glob_kappa_quadratic": r3(kappa(A["glob"][k], B["glob"][k], "quadratic")),
        "global_not_normal_kappa": r3(kappa((A["glob"][k] > 0).astype(int),
                                            (B["glob"][k] > 0).astype(int))),
        "slow_kappa_quadratic": r3(kappa(A["slow"][k], B["slow"][k], "quadratic")),
        "pdr_icc_readers": r3(icc_a1(A["pdr"][k], B["pdr"][k])),
        "R1_global_auc_impaired_vs_CN": r3(auc(key["impaired"][k], A["glob"][k])),
        "pdr_icc_est_R1": r3(icc_a1(est["pdr_frequency"][k], A["pdr"][k])),
        "pdr_icc_est_R2": r3(icc_a1(est["pdr_frequency"][k], B["pdr"][k])),
        "consensus_spearman": {}}
    kk = np.flatnonzero(k.values)
    for item, conc in [("slow", "slowing_ratio"), ("slow", "theta_alpha_ratio"),
                       ("slow", "rel_theta"), ("pmax", "amp_posterior_maximality"),
                       ("pmax", "posterior_maximality"), ("wf", "pdr_prominence"),
                       ("art", "artefact_proxy"), ("drow", "theta_alpha_ratio")]:
        cons = ((A[item] + B[item]) / 2).values[kk]
        x = est[conc].values[kk]
        ci = boot(lambda ix: rho(cons[ix], x[ix])[0], len(kk))
        res["without_incoherent_R1_rows"]["consensus_spearman"][f"{item}~{conc}"] = {
            "spearman": r3(rho(cons, x)[0]), "ci": [r3(v) for v in ci]}
    # paired difference between the two posterior-maximality forms against the same grade
    cons = ((A["pmax"] + B["pmax"]) / 2).values
    fa, fr = est["amp_posterior_maximality"].values, est["posterior_maximality"].values
    dfn = lambda ix: rho(cons[ix], fa[ix])[0] - rho(cons[ix], fr[ix])[0]
    res["pmax_amp_minus_relative"] = {"diff": r3(dfn(np.arange(n))),
                                      "ci": [r3(v) for v in boot(dfn, n)]}
    res["pdr_rank_agreement"] = {"inter_reader": ir["pdr_frequency"]["spearman"],
                                 "est_R1": re_["pdr_frequency_vs_R1"]["spearman"],
                                 "est_R2": re_["pdr_frequency_vs_R2"]["spearman"]}

    # per-recording table for the supplement
    tab = pd.concat([key[["blind_id", "cohort", "stratum", "age", "model_score"]],
                     A.add_prefix("R1_")[["R1_pdr", "R1_pdr_absent", "R1_wf", "R1_pmax", "R1_slow",
                                          "R1_art", "R1_drow", "R1_glob"]],
                     B.add_prefix("R2_")[["R2_pdr", "R2_pdr_absent", "R2_wf", "R2_pmax", "R2_slow",
                                          "R2_art", "R2_drow", "R2_glob"]],
                     est[["pdr_frequency", "pdr_prominence", "pdr_missing_frac", "rel_theta",
                          "theta_alpha_ratio", "slowing_ratio", "posterior_maximality", "amp_posterior_maximality",
                          "asymmetry_alpha", "amp_asymmetry_alpha", "artefact_proxy"]]], axis=1)
    tab.to_csv(OUT.with_name("reader_study_per_recording.csv"), index=False)

    from src.utils.provenance import write_result
    write_result(OUT, res, script=Path(__file__).name)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
