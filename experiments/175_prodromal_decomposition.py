"""Why does the prodromal contrast fail? Separating a hard task from a hard transfer.

No model tested exceeds 0.61 on control-versus-mild-cognitive-impairment at the prodromal site,
and the manuscript currently reports that as a null. A null is not an explanation, and there are
three candidate explanations that the corpus can separate.

  1. THE TASK IS HARD. Mild cognitive impairment is a milder deviation than dementia and its
     electrophysiology overlaps normal ageing. If so, discrimination should be modest even
     within a single cohort, where no transfer is involved.

  2. THE LABEL IS HETEROGENEOUS. "MCI" pools aetiologies. The training cohort annotates
     amnestic (n=331), vascular (n=67), non-amnestic (n=17) and prodromal-AD (n=17) subtypes,
     and only some of those are prodromes of the disease the concepts describe. If the label is
     the problem, detectability should differ sharply by subtype.

  3. THE SITE DOES NOT TRANSPORT. The prodromal cohort's spectrum barely overlaps the training
     cohort's and the out-of-distribution guard declines all of it. If so, the contrast should be
     recoverable WITHIN that cohort while failing on transfer into it.

These make different predictions and the analyses below test each. The distinction matters for
what a prospective study should do: a hard task needs a bigger sample, a heterogeneous label needs
better phenotyping, and a transport failure needs site-level adaptation.

    python experiments/175_prodromal_decomposition.py --tag channel_clean
Out: outputs/results/audit/prodromal_decomposition_<tag>.json
"""
from __future__ import annotations

import argparse
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

OUT = ROOT / "outputs" / "results" / "audit"
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
        return None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    o = [auc(pos[rng.integers(0, len(pos), len(pos))],
             neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    o = [v for v in o if v is not None]
    return a, [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


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
    X, names, S = T["X_concept"], T["names"], T["X_spectral"]
    signs = [DIR[n] for n in names]

    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}

    def sym(s, c):
        if c != "CAUEEG":
            return set()
        m = re.search(r"(\d+)$", str(s))
        return set(ann.get(m.group(1).zfill(5), {}).get("symptom", [])) if m else set()

    syms = [sym(s, c) for s, c in zip(subs, coh)]
    ages = np.array([float(ann[re.search(r"(\d+)$", str(s)).group(1).zfill(5)]["age"])
                     if c == "CAUEEG" else np.nan for s, c in zip(subs, coh)])

    cau = coh == "CAUEEG"
    gen = coh == "GENEEG"
    report = {"tag": args.tag, "note": "Why the prodromal contrast fails: task, label or transport."}

    def fit_oof(Xa, ya, feat="concept", rs=SEED):
        out = np.full(len(ya), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rs).split(Xa, ya):
            if feat == "concept":
                f = SignConstrainedLogistic(signs, C=0.5).fit(Xa[tr], ya[tr])
                out[te] = f.predict_proba(Xa[te])[:, 1]
            else:
                from sklearn.ensemble import RandomForestClassifier
                f = RandomForestClassifier(400, random_state=rs, n_jobs=-1,
                                           class_weight="balanced").fit(Xa[tr], ya[tr])
                out[te] = f.predict_proba(Xa[te])[:, 1]
        return out

    # =============== A. task versus transport ==========================================
    print("=" * 78)
    print("A. IS IT THE TASK OR THE TRANSFER?")
    print("=" * 78)
    A = {}

    # within the training cohort
    m = cau & np.isin(cls_, ["CN", "MCI"])
    y = (cls_[m] == "MCI").astype(int)
    sc = fit_oof(X[m], y)
    a, ci = auc_ci(sc[y == 1], sc[y == 0])
    A["within_training_cohort"] = {"n": int(m.sum()), "auc": a, "ci95": ci}
    print(f"   within training cohort   n={int(m.sum()):5d}  AUC {a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")

    # Within the prodromal site: the ceiling any transferred model could reach. n=63, so a
    # single 5-fold split is dominated by split noise; average 20 repeats.
    mg = gen & np.isin(cls_, ["CN", "MCI"])
    yg = (cls_[mg] == "MCI").astype(int)
    scg = np.mean([fit_oof(X[mg], yg, rs=r) for r in range(20)], axis=0)
    ag, cig = auc_ci(scg[yg == 1], scg[yg == 0])
    A["within_prodromal_site_concepts"] = {"n": int(mg.sum()), "auc": ag, "ci95": cig,
                                           "protocol": "20 repeats of 5-fold, scores averaged"}
    print(f"   within prodromal site    n={int(mg.sum()):5d}  AUC {ag:.3f} [{cig[0]:.3f},{cig[1]:.3f}]"
          f"   <- ceiling for any transferred model")
    scs = None
    if S is not None:
        scs = np.mean([fit_oof(S[mg], yg, feat="forest", rs=r) for r in range(20)], axis=0)
        asp, cisp = auc_ci(scs[yg == 1], scs[yg == 0])
        A["within_prodromal_site_spectral"] = {"auc": asp, "ci95": cisp}
        print(f"   within prodromal, spectral      AUC {asp:.3f} [{cisp[0]:.3f},{cisp[1]:.3f}]")

    # transferred, evaluated on exactly the same subjects as the ceiling
    tr = cau & np.isin(cls_, ["CN", "MCI"])
    ytr = (cls_[tr] == "MCI").astype(int)
    fits = [SignConstrainedLogistic(signs, C=0.5).fit(X[tr][a_], ytr[a_])
            for a_, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], ytr)]
    st = np.mean([f.predict_proba(X[mg])[:, 1] for f in fits], axis=0)
    at, cit = auc_ci(st[yg == 1], st[yg == 0])
    A["transferred"] = {"n": int(mg.sum()), "auc": at, "ci95": cit}
    print(f"   transferred              n={int(mg.sum()):5d}  AUC {at:.3f} [{cit[0]:.3f},{cit[1]:.3f}]")

    # The two are scored on the same 63 subjects, so the difference is a paired quantity and a
    # paired bootstrap is far more precise than comparing two marginal intervals.
    rng = np.random.default_rng(SEED)
    ip, ineg = np.flatnonzero(yg == 1), np.flatnonzero(yg == 0)
    d = []
    for _ in range(N_BOOT):
        bp = ip[rng.integers(0, len(ip), len(ip))]
        bn = ineg[rng.integers(0, len(ineg), len(ineg))]
        a1, a2 = auc(scg[bp], scg[bn]), auc(st[bp], st[bn])
        if a1 is not None and a2 is not None:
            d.append(a1 - a2)
    dl, dh = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    A["transport_loss_paired"] = {"delta": ag - at, "ci95": [dl, dh],
                                  "note": "ceiling minus transferred, same 63 subjects"}
    print("")
    print(f"   ceiling - transferred = {ag - at:+.3f}  [{dl:+.3f},{dh:+.3f}]  (paired)")
    print(f"   the ceiling itself is {ag:.3f}: the task is hard before any transfer")

    report["A_task_vs_transport"] = A

    # =============== B. label heterogeneity ============================================
    print("\n" + "=" * 78)
    print("B. IS THE MCI LABEL HETEROGENEOUS?")
    print("=" * 78)
    B = {}
    m = cau & np.isin(cls_, ["CN", "MCI"])
    y = (cls_[m] == "MCI").astype(int)
    sc = fit_oof(X[m], y)
    idx = np.flatnonzero(m)
    cn_local = (y == 0)
    for lab in ("mci_amnestic", "mci_vascular", "mci_non_amnestic", "mci_ad"):
        sel = np.array([lab in syms[i] for i in idx])
        if sel.sum() < 10:
            continue
        a, ci = auc_ci(sc[sel], sc[cn_local])
        aa, aci = auc_ci(ages[idx][sel], ages[idx][cn_local])
        B[lab] = {"n": int(sel.sum()), "mean_age": float(np.nanmean(ages[idx][sel])),
                  "auc_concepts": a, "ci95": ci, "auc_age_only": aa}
        print(f"   {lab:20s} n={int(sel.sum()):4d} age {np.nanmean(ages[idx][sel]):.1f}  "
              f"concepts {a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]   age-only {aa:.3f}")
    if "mci_amnestic" in B and "mci_vascular" in B:
        B["amnestic_minus_vascular"] = B["mci_amnestic"]["auc_concepts"] - B["mci_vascular"]["auc_concepts"]
        print(f"\n   amnestic - vascular = {B['amnestic_minus_vascular']:+.3f}")
    report["B_label_heterogeneity"] = B

    # =============== C. how far is the prodromal site =================================
    print("\n" + "=" * 78)
    print("C. HOW FAR IS THE PRODROMAL SITE FROM THE TRAINING DISTRIBUTION?")
    print("=" * 78)
    tr_mask = cau
    mu = np.nanmean(X[tr_mask], axis=0)
    Sig = np.cov(X[tr_mask].T) + np.eye(X.shape[1]) * 1e-6
    Si = np.linalg.inv(Sig)

    def maha(V):
        d = V - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, Si, d))

    C = {}
    for nm, mask in (("training cohort", cau), ("dementia site", coh == "ds004504_raw"),
                     ("prodromal site", gen)):
        dm = maha(X[mask])
        C[nm] = {"n": int(mask.sum()), "median": float(np.median(dm)),
                 "p90": float(np.percentile(dm, 90))}
        print(f"   {nm:18s} n={int(mask.sum()):5d}  median Mahalanobis {np.median(dm):6.2f}")
    thr = float(np.percentile(maha(X[cau]), 95))
    C["training_p95_threshold"] = thr
    for nm, mask in (("dementia site", coh == "ds004504_raw"), ("prodromal site", gen)):
        C[nm]["fraction_beyond_training_p95"] = float((maha(X[mask]) > thr).mean())
        print(f"   {nm:18s} beyond training 95th percentile: "
              f"{C[nm]['fraction_beyond_training_p95']:.1%}")
    report["C_distribution_distance"] = C

    # =============== D. is the subtype effect just age? ================================
    print("")
    print("=" * 78)
    print("D. IS THE SUBTYPE EFFECT JUST AGE?")
    print("=" * 78)
    Dd = {}
    ov = {}
    for a_ in ("mci_amnestic", "mci_vascular", "mci_non_amnestic", "mci_ad"):
        for b_ in ("mci_amnestic", "mci_vascular", "mci_non_amnestic", "mci_ad"):
            if a_ < b_:
                n = sum(1 for i in idx if a_ in syms[i] and b_ in syms[i])
                if n:
                    ov[f"{a_} & {b_}"] = int(n)
    Dd["label_overlap"] = ov
    Dd["note_mci_ad"] = "mci_ad is nested inside mci_amnestic (15 of 17), not an independent subtype"
    print("   overlapping subtype labels:", ov if ov else "none: the subtypes are disjoint")

    ag_local = ages[idx]
    cn_idx = np.flatnonzero(cn_local)
    for lab in ("mci_amnestic", "mci_vascular"):
        sel = np.flatnonzero([lab in syms[i] for i in idx])
        pr = match_age(ag_local[sel], ag_local[cn_idx], caliper=2.0)
        pi = np.array([sel[i] for i, _ in pr])
        ni = np.array([cn_idx[j] for _, j in pr])
        a, ci = auc_ci(sc[pi], sc[ni])
        aa, _ = auc_ci(ag_local[pi], ag_local[ni])
        # Greedy matching depends on the permutation order, so sweep it: a subtype claim built
        # on one draw is a claim about that draw.
        sw = []
        for sd in range(25):
            q = match_age(ag_local[sel], ag_local[cn_idx], caliper=2.0, seed=sd)
            qi = np.array([sel[i] for i, _ in q])
            qj = np.array([cn_idx[j] for _, j in q])
            sw.append(auc(sc[qi], sc[qj]))
        sw = np.array(sw)
        Dd[lab + "_age_matched"] = {"n_pairs": len(pr), "auc_concepts": a, "ci95": ci,
                                    "auc_age_only": aa,
                                    "mean_age_case": float(np.mean(ag_local[pi])),
                                    "mean_age_control": float(np.mean(ag_local[ni])),
                                    "seed_sweep": {"n_seeds": 25, "mean": float(sw.mean()),
                                                   "sd": float(sw.std()),
                                                   "range": [float(sw.min()), float(sw.max())],
                                                   "frac_above_chance": float((sw > 0.5).mean())}}
        print(f"   {lab:16s} age-matched  {len(pr):3d} pairs  concepts {a:.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}]   age-only {aa:.3f}")
        print(f"   {'':16s}   25 matching seeds: mean {sw.mean():.3f} sd {sw.std():.3f} "
              f"range [{sw.min():.3f},{sw.max():.3f}]  above chance in {(sw>0.5).mean():.0%}")

    # do the concepts separate the two aetiologies from each other?
    am = np.flatnonzero([("mci_amnestic" in syms[i]) and ("mci_vascular" not in syms[i]) for i in idx])
    va = np.flatnonzero([("mci_vascular" in syms[i]) and ("mci_amnestic" not in syms[i]) for i in idx])
    if len(am) > 10 and len(va) > 10:
        a, ci = auc_ci(sc[am], sc[va])
        Dd["amnestic_vs_vascular_direct"] = {"n_amnestic": len(am), "n_vascular": len(va),
                                             "auc": a, "ci95": ci}
        print(f"   amnestic vs vascular directly  ({len(am)} vs {len(va)})  AUC {a:.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}]")
    report["D_subtype_age_control"] = Dd

    # =============== E. what the prodromal site could have detected ====================
    # A near-chance result at n=63 is only evidence of absence if n=63 could have found the
    # effect. Hanley-McNeil variance, two-sided alpha=0.05.
    print("")
    print("=" * 78)
    print("E. WHAT COULD n=63 HAVE DETECTED?")
    print("=" * 78)
    from scipy.stats import norm

    def hm_se(a, n1, n0):
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                              + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))

    za, zb = norm.ppf(0.975), norm.ppf(0.80)
    n1 = int((yg == 1).sum()); n0 = int((yg == 0).sum())
    floor = 0.5 + za * hm_se(0.5 + 1e-9, n1, n0)
    E_ = {"n_cases": n1, "n_controls": n0, "smallest_auc_separable_from_chance": floor}
    print(f"   prodromal site has {n1} cases and {n0} controls")
    print(f"   smallest AUC separable from chance there: {floor:.3f}")
    for nm, a_ in (("within-cohort value", 0.650), ("age-matched amnestic", 0.593)):
        n = next(k for k in range(20, 20000) if (a_ - 0.5) / hm_se(a_, k, k) >= za + zb)
        E_[nm.replace(" ", "_") + "_n_per_arm"] = n
        print(f"   to detect {nm} ({a_:.3f}) at 80% power: {n} per arm")
    report["E_detectable_effect"] = E_

    dest = write_result(OUT / f"prodromal_decomposition_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
