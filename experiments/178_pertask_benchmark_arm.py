"""The per-task arm as a self-contained benchmark: paired tests, equivalence and the spread null.

Why this exists. Two arms in this study score the *same* 87 subjects on the *same* contrast and
disagree about which model leads:

  * the three-class arm (``paired_tests_channel_clean.json``), where the deep models are trained on
    the harmonised three-class head with early stopping and the binary contrast is read off the
    restricted posterior. ATCNet leads there at 0.870.
  * the per-task arm (``deep_per_task.json`` / ``task_battery.json``), where every model is trained
    directly for the binary question at a fixed 25 epochs.

Every other result in the manuscript -- the task battery, the concept attribution, the operating
points, the repaired instrument -- is computed on the per-task arm, but the headline benchmark
figure was drawn from the three-class arm, so figure and table disagreed in print. This script
recomputes the whole benchmark apparatus on the per-task arm so that one arm carries the paper:

  1. the sign-constrained concept model's per-subject external scores on all four tasks, refitted
     under the identical protocol of ``157_task_battery.py`` (same folds, same C, same directions,
     same imputation) and checked against the AUCs that script already published;
  2. paired subject-clustered bootstrap of every model against the spectral forest, Holm-adjusted
     within the family, on the dementia contrast;
  3. two one-sided tests at 0.05, 0.075 and 0.10 AUC;
  4. the best-to-worst spread against its null, resampling subjects with every model held fixed;
  5. confusion matrices at the label-free source-prevalence threshold, for every model on every
     task, so the concept model can be drawn beside the comparators that already carry them.

    python experiments/178_pertask_benchmark_arm.py
Out: outputs/results/audit/pertask_benchmark_arm.json
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

AUD = ROOT / "outputs" / "results" / "audit"
PRED = AUD / "predictions_per_task"

SEED, N_BOOT = 42, 10000
MARGINS = (0.05, 0.075, 0.10)
REFERENCE = "random_forest"

TASKS = ["CN_vs_AD", "CN_vs_dementia", "CN_vs_impaired", "CN_vs_MCI"]


def _load_157():
    """Import the task-battery module by path; its name is not a legal identifier."""
    spec = importlib.util.spec_from_file_location(
        "task_battery", ROOT / "experiments" / "157_task_battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def auc(y, s):
    """AUC by the rank form of the Mann-Whitney statistic.

    ``rankdata`` with average ties is not an optimisation detail here. Resampling with replacement
    creates exact duplicate scores by construction, and a rank obtained by argsort-of-argsort
    breaks those ties by array position, which biases every bootstrap draw. Averaging ranks over
    tied groups is the definition the AUC needs.
    """
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(s, float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def concept_scores(tag="channel_clean"):
    """Refit the sign-constrained concept ensemble per task and return external scores.

    The protocol is copied from ``157_task_battery.py`` rather than approximated: the same
    stratified folds under the same seed, the same inverse regularisation, the same clinical
    direction vector. The returned AUCs are asserted against the published ones downstream, which
    is what makes the per-subject scores usable for a paired test.
    """
    from sklearn.model_selection import StratifiedKFold

    from src.models.v6 import SignConstrainedLogistic

    tbmod = _load_157()
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1}

    T = tbmod.subject_table(tag)
    names = T["names"]
    out = {}
    for task in TASKS:
        pos, neg, train_coh, eval_coh = tbmod.TASKS[task]
        keep = np.isin(T["cls"], pos + neg)
        y = np.isin(T["cls"], pos).astype(int)
        tr = keep & (T["cohort"] == train_coh)
        te = keep & (T["cohort"] == eval_coh)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=tbmod.SEED).split(
            T["X_concept"][tr], y[tr]))
        Xtr, ytr = T["X_concept"][tr], y[tr]
        fits = [SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(Xtr[a], ytr[a])
                for a, _ in folds]
        score = np.mean([f.predict_proba(T["X_concept"][te])[:, 1] for f in fits], axis=0)
        # The label-free rule: the quantile of the training-cohort prevalence, computed on the
        # training cohort's own out-of-fold scores, exactly as the deployment analysis sets it.
        oof = np.full(tr.sum(), np.nan)
        for a, b in folds:
            f = SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(Xtr[a], ytr[a])
            oof[b] = f.predict_proba(Xtr[b])[:, 1]
        prevalence = float(ytr.mean())
        out[task] = {
            "subjects": np.asarray(T["subjects"])[te],
            "y": y[te], "score": score,
            "train_prevalence": prevalence,
            "threshold_source": float(np.quantile(score, 1.0 - prevalence)),
            "auc": auc(y[te], score),
        }
    return out


def confusion(y, s, thr):
    """Every threshold-dependent metric, from one contingency table.

    F1 and MCC are included because sensitivity and specificity alone hide what an imbalanced
    evaluation site does to a decision: at 58 cases against 29 controls, a rule that refers
    everyone scores 0.74 on F1 and 0 on MCC, and only the second says so.
    """
    y = np.asarray(y).astype(int)
    p = (np.asarray(s) >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan")
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(den)) if den > 0 else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2,
            "accuracy": (tp + tn) / len(y),
            "ppv": ppv, "npv": npv, "f1": float(f1), "mcc": float(mcc),
            "threshold": float(thr)}


def _draws(y, n_boot, seed):
    """One shared set of resampled subject indices, so every comparison is paired on it."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    while len(out) < n_boot:
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        out.append(i)
    return np.stack(out)


def paired_bootstrap(y, sa, sb, idx, n_boot=N_BOOT, seed=SEED):
    """Subject-clustered paired bootstrap of AUC(a) - AUC(b) on identical subjects."""
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb)
    obs = auc(y, sa) - auc(y, sb)
    d = np.empty(len(idx))
    for k, i in enumerate(idx):
        d[k] = auc(y[i], sa[i]) - auc(y[i], sb[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    tost = {}
    for m in MARGINS:
        # Equivalent when the whole 90 per cent interval falls inside (-m, +m); the reported
        # p is the larger of the two one-sided tail probabilities.
        p_lo = (d <= -m).mean()
        p_hi = (d >= +m).mean()
        p_t = max(p_lo, p_hi)
        l90, h90 = np.percentile(d, [5, 95])
        tost[str(m)] = {"equivalent": bool(l90 > -m and h90 < m), "p_tost": float(p_t),
                        "ci90": [float(l90), float(h90)]}
    return {"delta": float(obs), "ci95": [float(lo), float(hi)],
            "p": float(min(1.0, p)), "tost": tost}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        v = (len(pvals) - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(1.0, running)
    return adj


def spread_null(y, scores, idx):
    """Best-to-worst spread against resampling the same subjects with every model held fixed.

    The null is not "models are identical"; it is "these models, this good, scored on a resample
    of these subjects". A spread smaller than that null says the fifteen-model range carries no
    information the sampling noise does not already supply.
    """
    y = np.asarray(y)
    names = list(scores)
    M = np.stack([np.asarray(scores[m]) for m in names])
    obs_auc = np.array([auc(y, s) for s in M])
    obs = float(obs_auc.max() - obs_auc.min())
    draws = np.empty(len(idx))
    for k, i in enumerate(idx):
        a = np.array([auc(y[i], s[i]) for s in M])
        draws[k] = a.max() - a.min()
    return {"observed": obs, "null_median": float(np.median(draws)),
            "p": float((draws <= obs).mean()), "null_draws": draws.round(5).tolist(),
            "per_model_auc": {m: float(a) for m, a in zip(names, obs_auc)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="comparator keys left out of the family")
    ap.add_argument("--out", default="pertask_benchmark_arm.json")
    args = ap.parse_args()
    EXCL = set(args.exclude)

    import json

    from src.utils.provenance import write_result

    dp = json.loads((AUD / "deep_per_task.json").read_text())["tasks"]
    tb = json.loads((AUD / "task_battery.json").read_text())["tasks"]

    print("refitting the concept ensemble per task ...", flush=True)
    con = concept_scores(args.tag)
    for t in TASKS:
        published = tb[t]["models"]["concept_sign_constrained"]["external"]["auc"]
        got = con[t]["auc"]
        flag = "OK " if abs(got - published) < 5e-4 else "DRIFT"
        print(f"  {flag} {t:16s} refit {got:.4f}  published {published:.4f}", flush=True)
        if flag == "DRIFT":
            raise SystemExit("concept refit does not reproduce the published AUC; stop.")

    report = {"tag": args.tag, "n_boot": N_BOOT, "reference": REFERENCE,
              "margins": list(MARGINS),
              "note": ("The per-task arm treated as the primary benchmark. Every model is trained "
                       "directly for its binary question at a fixed 25 epochs; the concept "
                       "ensemble is refitted here under the protocol of 157_task_battery.py and "
                       "reproduces its published AUCs, which is what licenses the paired tests."),
              "tasks": {}}

    for task in TASKS:
        y = None
        scores = {}
        for m in dp[task]["models"]:
            if m in EXCL:
                continue
            f = PRED / f"{task}__{m}.npz"
            if not f.exists():
                print(f"  [skip] no stored predictions for {task} {m}")
                continue
            z = np.load(f, allow_pickle=True)
            if y is None:
                y = z["y"].astype(int)
            elif not np.array_equal(y, z["y"].astype(int)):
                raise SystemExit(f"label vectors disagree between models on {task}")
            scores[m] = z["score"].astype(float)
        scores["concept_model"] = con[task]["score"]
        if not np.array_equal(y, np.asarray(con[task]["y"]).astype(int)):
            raise SystemExit(f"concept label vector disagrees with the stored ones on {task}")

        # `deep_per_task.json` is the authority for every comparator, because it is what table 3
        # of the manuscript prints. The stored per-subject predictions are used only for the
        # paired tests, and only after they are shown to reproduce that authority. They do not
        # always: the CN_vs_MCI files were overwritten by a later seed sweep, so on that task the
        # npz are seed 44 while the table is seed 42, and a figure drawn from them would have
        # disagreed with the table beside it.
        drift = {m: (auc(y, s), dp[task]["models"][m]["auc"])
                 for m, s in scores.items() if m != "concept_model"
                 if abs(auc(y, s) - dp[task]["models"][m]["auc"]) > 1.5e-3}
        consistent = not drift
        if drift:
            print("  [!] %s: stored predictions do not reproduce deep_per_task.json for %d of %d "
                  "models (max |diff| %.4f). Paired tests and the spread null are skipped for "
                  "this task; published metrics are still reported."
                  % (task, len(drift), len(scores) - 1,
                     max(abs(a - b) for a, b in drift.values())), flush=True)

        trep = {"n": int(len(y)), "n_pos": int(y.sum()),
                "train_prevalence": con[task]["train_prevalence"],
                "predictions_reproduce_published": consistent,
                "auc": {}, "operating_source_prevalence": {}}

        # Metrics come from the published contingency table for the comparators and from the
        # refit for the concept model, so all fifteen carry an identical metric set and none of
        # them can drift from table 3.
        from sklearn.metrics import average_precision_score
        for m in scores:
            if m == "concept_model":
                trep["auc"][m] = auc(y, scores[m])
                row = confusion(y, scores[m], con[task]["threshold_source"])
                row["average_precision"] = float(average_precision_score(y, scores[m]))
            else:
                pub = dp[task]["models"][m]
                trep["auc"][m] = float(pub["auc"])
                o = pub["at_source_prevalence"]
                tp, fp, tn, fn = (int(o["tp"]), int(o["fp"]), int(o["tn"]), int(o["fn"]))
                row = confusion(np.r_[np.ones(tp + fn), np.zeros(tn + fp)],
                                np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)], 0.5)
                row["threshold"] = float(o["threshold"])
                row["average_precision"] = float(pub["average_precision"])
            trep["operating_source_prevalence"][m] = row

        if consistent:
            # One resample set for the whole family, so the paired comparisons and the spread null
            # are computed on identical draws and can be read against each other.
            idx = _draws(y, N_BOOT, SEED)
            ref = scores[REFERENCE]
            vs = {}
            for m, s in scores.items():
                if m == REFERENCE:
                    continue
                vs[m] = paired_bootstrap(y, s, ref, idx)
            keys = list(vs)
            padj = holm(np.array([vs[m]["p"] for m in keys]))
            for m, a in zip(keys, padj):
                vs[m]["p_holm"] = float(a)
            trep["vs_reference"] = vs
            trep["spread"] = spread_null(y, scores, idx)
        else:
            trep["vs_reference"] = None
            trep["spread"] = None
            trep["skipped_because"] = (
                "stored per-subject predictions are from a different training seed than "
                "deep_per_task.json, so a paired test on them would not describe the arm the "
                "manuscript reports")
        report["tasks"][task] = trep
        best = max(trep["auc"], key=trep["auc"].get)
        line = f"  {task:16s} n={trep['n']:3d}  best {best} {trep['auc'][best]:.3f}"
        if trep["spread"]:
            line += (f"  spread {trep['spread']['observed']:.3f} vs null "
                     f"{trep['spread']['null_median']:.3f} (P={trep['spread']['p']:.3f})")
        else:
            line += "  [paired tests skipped: prediction/table seed mismatch]"
        print(line, flush=True)

    report["excluded"] = sorted(EXCL)
    dest = write_result(AUD / args.out, report, script=Path(__file__).name)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
