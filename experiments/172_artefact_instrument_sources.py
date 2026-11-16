"""What does the artefact instrument actually index?

The battery's discriminant check sets each concept's criterion correlation against its correlation
with an artefact instrument, defined as mean relative 30-45 Hz power. Two pre-specified tests were
run to establish that the instrument measures scalp electromyogram and neither supported it in the
training cohort: the ICLabel correlation reached only rho=+0.311 there against a pre-specified
threshold of 0.4 (`123_ica_muscle_check.py`), and the myogenic topography predicted from the
electromyographic literature is absent across the whole cohort (`117_artefact_proxy_validation.py`).

The manuscript therefore reports the instrument as tracking high-frequency non-cortical variance of
unresolved origin, and names two untested candidates: residual ocular activity and drowsiness. This
script tests both, and re-tests muscle on the full corpus rather than the 90-recording subsample the
ICLabel check could afford.

Three quantities are available for every one of the 1,296 analysed recordings, from the preprocessing
sidecars written when the harmonised cache was built:

  * `variance_removed`  -- the share of variance carried by the components ICA removed. Ocular,
                           cardiac, line and channel components are removed; muscle is retained by
                           design. Since the composition audit puts ocular at 58.5 per cent of
                           training-cohort variance against 0.6 per cent for muscle, this is an
                           ocular burden measure in all but name.
  * `ica_labels`        -- component COUNTS by ICLabel class, including eye blink and muscle.
  * window `start`      -- position of each epoch within its recording.

The tests, and what each would show:

  P1 OCULAR. If the instrument indexes residual ocular activity that survived component removal,
     it should correlate with how much ocular activity the recording carried in the first place.
     A recording needing more ocular correction plausibly leaves more behind.

  P2 MUSCLE. The n=90 ICLabel test is repeated at n=1,296 using component counts rather than
     variance shares. Counts are coarser but available for everyone, which buys back the power the
     subsample lost. A muscle association that was real but underpowered at 90 should appear here.

  P3 DROWSINESS. Vigilance declines through a resting recording, so on average later epochs are
     drowsier than earlier ones. If the instrument tracks drowsiness it should drift with epoch
     position WITHIN a subject. This is tested within-subject, so it is not confounded by anything
     that differs between subjects, and it is not circular: epoch position is a clock, not a
     spectral quantity.

  P3 is a weak test of a strong hypothesis -- vigilance decline is only one component of drowsiness
  and 96 s of retained recording spans a short window of it -- so a null there does not exonerate
  drowsiness. It is reported for what it is.

    python experiments/172_artefact_instrument_sources.py --tag channel_clean
Out: outputs/results/audit/artefact_instrument_sources_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result       # noqa: E402
from src.data.feature_table import check_table      # noqa: E402

CACHE = ROOT / "outputs" / "harmonised"
FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"


def spearman_ci(a, b, n_boot=2000, seed=42):
    """Spearman rho with a percentile bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return None, None, 0
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)          # noqa: E731

    def rho(x, y):
        return float(np.corrcoef(rank(x), rank(y))[0, 1])

    r = rho(a, b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(np.unique(a[i])) < 3 or len(np.unique(b[i])) < 3:
            continue
        boots.append(rho(a[i], b[i]))
    if not boots:
        return r, None, len(a)
    return r, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], len(a)


def auc_ci(x, present, n_boot=2000, seed=42):
    """Rank AUC of x separating present from absent, with a bootstrap interval.

    Half the corpus carries no muscle component at all, so a Spearman against the component share
    is dominated by one tie block at zero and its bootstrap misbehaves -- the point estimate can
    fall outside the percentile interval, which is how the problem announces itself. A two-group
    rank statistic is the honest summary of a variable that is mostly zero: it asks whether
    recordings with any such component carry more of the instrument's quantity than those without.
    """
    x, present = np.asarray(x, float), np.asarray(present, bool)
    ok = np.isfinite(x)
    x, present = x[ok], present[ok]
    if present.sum() < 5 or (~present).sum() < 5:
        return None, None, 0, 0

    def a(v, p):
        r = np.argsort(np.argsort(v)) + 1.0
        n1, n0 = p.sum(), (~p).sum()
        return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    val = a(x, present)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if present[i].sum() < 3 or (~present[i]).sum() < 3:
            continue
        boots.append(a(x[i], present[i]))
    ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
          if boots else None)
    return val, ci, int(present.sum()), int((~present).sum())


def fmt(r, ci):
    if r is None:
        return "     n/a"
    s = f"{r:+.3f}"
    if ci:
        s += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        s += "  *" if (ci[0] > 0 or ci[1] < 0) else "   "
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import CONCEPT_NAMES, aggregate_subject

    d = FEAT / args.tag
    check_table(d)
    C_win = np.load(d / "window_concepts.npy")
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, starts = meta["groups"], meta["start"]
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])

    j_art = CONCEPT_NAMES.index("artefact_proxy")
    art = C[:, j_art]
    flagged = ["aperiodic_exponent", "spectral_edge", "spectral_entropy"]
    controls = ["pdr_frequency", "theta_alpha_ratio"]

    # ---- sidecar quantities, per subject ------------------------------------------------
    side = {}
    for f in CACHE.glob("*.json"):
        try:
            m = json.loads(f.read_text())
        except Exception:                                             # noqa: BLE001
            continue
        q = m.get("qc") or {}
        lab = q.get("ica_labels") or {}
        ncomp = max(sum(lab.values()), 1)
        side[str(m.get("subject"))] = {
            "variance_removed": q.get("variance_removed"),
            "eye_frac": lab.get("eye blink", 0) / ncomp,
            "muscle_frac": lab.get("muscle artifact", 0) / ncomp,
            "eye_n": lab.get("eye blink", 0),
            "muscle_n": lab.get("muscle artifact", 0),
        }

    g = lambda k: np.array([                                          # noqa: E731
        (side.get(str(s)) or {}).get(k, np.nan) for s in subs], dtype=float)
    var_rm, eye_f, mus_f = g("variance_removed"), g("eye_frac"), g("muscle_frac")

    report = {
        "note": "What the artefact instrument (mean relative 30-45 Hz power) indexes. Tests the "
                "two candidates the manuscript names as untested -- residual ocular activity and "
                "drowsiness -- and repeats the muscle test at full corpus size.",
        "tag": args.tag,
        "n_subjects": int(len(subs)),
        "instrument": "artefact_proxy (mean relative 30-45 Hz power)",
        "tests": {},
    }

    print(f"corpus {args.tag}: {len(subs)} subjects")
    n_zero_mus = float((g("muscle_n") == 0).mean())
    n_zero_eye = float((g("eye_n") == 0).mean())
    report["tie_structure"] = {"fraction_with_no_muscle_component": n_zero_mus,
                               "fraction_with_no_eye_component": n_zero_eye}
    print(f"  {n_zero_mus:.1%} of recordings carry no muscle component, "
          f"{n_zero_eye:.1%} no eye component\n")

    print("P1/P2  subject-level associations of the instrument")
    print(f"  {'quantity':30s} {'vs instrument':>30s} {'n':>6s}")
    for nm, v in (("ocular: variance removed", var_rm),
                  ("ocular: eye component share", eye_f),
                  ("muscle: component share", mus_f)):
        r, ci, n = spearman_ci(art, v)
        report["tests"][nm] = {"instrument": {"rho": r, "ci95": ci, "n": n}}
        note = ""
        if ci and not (ci[0] <= r <= ci[1]):
            note = "  <- point estimate outside its own bootstrap interval; tie-dominated, " \
                   "read the two-group statistic instead"
            report["tests"][nm]["instrument"]["unreliable_tie_dominated"] = True
        print(f"  {nm:30s} {fmt(r, ci):>30s} {n:6d}{note}")

    # Tie-robust form: does the instrument separate recordings that carry such a component at all?
    print(f"\n  two-group rank AUC (component present vs absent)")
    for nm, cnt in (("muscle component present", g("muscle_n")),
                    ("eye component present", g("eye_n"))):
        v, ci, n1, n0 = auc_ci(art, cnt > 0)
        report["tests"][nm] = {"auc": v, "ci95": ci, "n_present": n1, "n_absent": n0}
        star = "  *" if (ci and (ci[0] > 0.5 or ci[1] < 0.5)) else "   "
        print(f"  {nm:30s} AUC {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{star} "
              f"present={n1} absent={n0}")

    print(f"\n  {'concept':22s} {'vs ocular(var_rm)':>28s} {'vs muscle(frac)':>28s}")
    for name in flagged + controls:
        v = C[:, CONCEPT_NAMES.index(name)]
        r1, c1, _ = spearman_ci(v, var_rm)
        r2, c2, _ = spearman_ci(v, mus_f)
        tag = "flagged" if name in flagged else "CONTROL"
        report["tests"].setdefault("per_concept", {})[name] = {
            "group": tag,
            "vs_ocular_variance_removed": {"rho": r1, "ci95": c1},
            "vs_muscle_component_share": {"rho": r2, "ci95": c2},
        }
        print(f"  {name:22s} {fmt(r1, c1):>28s} {fmt(r2, c2):>28s}  {tag}")

    # ---- P3 drowsiness: within-subject drift with epoch position -------------------------
    # Spearman of the instrument against epoch start time, computed inside each subject and
    # then averaged. A between-subject correlation would be confounded by recording length.
    # The instrument is a subject-level derived quantity and has no column in window_concepts.
    # Its window-level equivalent is mean relative gamma over the 19 channels, taken from the
    # independently written spectral block; check 5 puts the two at rho = 0.993, so this is the
    # same quantity computed per window rather than a substitute for it.
    S_win = np.load(d / "spectral.npy")
    if len(S_win) != len(C_win):
        raise SystemExit(f"row mismatch: {len(C_win)} concept rows vs {len(S_win)} spectral")
    gamma_cols = [c * 5 + 4 for c in range(19)]          # 19 channels x 5 bands, gamma last
    art_win = S_win[:, gamma_cols].mean(axis=1)

    per_sub = []
    for i in idxs:
        if len(i) < 8:
            continue
        w = art_win[i]
        t = starts[i].astype(float)
        ok = np.isfinite(w)
        if ok.sum() < 8 or len(np.unique(t[ok])) < 3:
            continue
        rank = lambda v: np.argsort(np.argsort(v)).astype(float)      # noqa: E731
        per_sub.append(float(np.corrcoef(rank(w[ok]), rank(t[ok]))[0, 1]))
    per_sub = np.array(per_sub, dtype=float)
    per_sub = per_sub[np.isfinite(per_sub)]
    rng = np.random.default_rng(42)
    boots = [float(np.mean(per_sub[rng.integers(0, len(per_sub), len(per_sub))]))
             for _ in range(2000)]
    mean_r = float(per_sub.mean())
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    frac_pos = float((per_sub > 0).mean())
    report["tests"]["drowsiness: within-subject drift with epoch position"] = {
        "mean_within_subject_rho": mean_r, "ci95": ci,
        "n_subjects": int(len(per_sub)), "fraction_positive": frac_pos,
        "caveat": "vigilance decline is one component of drowsiness and 96 s spans little of it; "
                  "a null here does not exonerate drowsiness",
    }
    print(f"\nP3  drowsiness proxy: instrument against epoch position, within subject")
    print(f"  mean rho {mean_r:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"over {len(per_sub)} subjects, {frac_pos:.1%} positive")

    # ---- verdict ------------------------------------------------------------------------
    oc = report["tests"]["ocular: variance removed"]["instrument"]
    mus_auc = report["tests"]["muscle component present"]
    eye_auc = report["tests"]["eye component present"]

    def sig(e):
        c = e.get("ci95")
        return bool(c and (c[0] > 0 or c[1] < 0))

    def sig_auc(e):
        c = e.get("ci95")
        return bool(c and (c[0] > 0.5 or c[1] < 0.5))

    verdict = []
    verdict.append(f"ocular, continuous (variance removed): rho={oc['rho']:+.3f}, "
                   f"{'excludes' if sig(oc) else 'covers'} zero")
    verdict.append(f"muscle, two-group: AUC={mus_auc['auc']:.3f}, "
                   f"{'excludes' if sig_auc(mus_auc) else 'covers'} 0.5")
    verdict.append(f"ocular, two-group: AUC={eye_auc['auc']:.3f}, "
                   f"{'excludes' if sig_auc(eye_auc) else 'covers'} 0.5")
    verdict.append(f"drowsiness (position drift): mean rho={mean_r:+.4f}, "
                   f"{'excludes' if (ci[0] > 0 or ci[1] < 0) else 'covers'} zero")
    report["verdict"] = verdict
    print("\nverdict:")
    for v in verdict:
        print("  " + v)

    dest = write_result(OUT / f"artefact_instrument_sources_{args.tag}.json", report,
                        script=Path(__file__).name)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
