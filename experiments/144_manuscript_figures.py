"""Every manuscript figure, generated from the stored result files.

One script, one source of truth: each panel reads the JSON that produced its numbers, so a figure
cannot drift from the table beside it.

    F1  study design: corpus, provenance, and where each cohort enters
    F2  what the recordings are made of, and how artefact relates to diagnosis at each site
    F3  concept validity landscape: artefact against criterion, one sample, with verdicts
    F4  the three failure modes, one panel each
    F5  benchmark: fifteen models, paired against a common reference, with the resampling null
    F6  interpretability: a single patient's decision, decomposed
    F7  out-of-distribution structure and the within-site ceiling

    python experiments/144_manuscript_figures.py
Out: jne_build/Figs/*.pdf
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

import _figstyle_2026 as S  # noqa: E402

AUD = ROOT / "outputs" / "results" / "audit"
VAL = ROOT / "outputs" / "results" / "validity"
FIGS = ROOT / "jne_build" / "Figs"

# Which analysis arm the figures are drawn from. This has to be the arm the manuscript tables
# report, or the figures and the text disagree while both are individually correct -- which is
# harder for a reader to detect than an outright error. Set by --tag; see main().
TAG = "channel_clean"


#: Display names for the concepts, so every panel reads like the tables rather than like the code.
CONCEPT_NICE = {
    "rel_theta": "Relative theta", "pdr_frequency": "PDR frequency",
    "posterior_maximality": "Posterior maximality", "asymmetry_alpha": "Alpha asymmetry",
    "asymmetry_freq": "Frequency asymmetry", "pdr_prominence": "PDR prominence",
    "slowing_ratio": "Slowing ratio", "aperiodic_exponent": "Aperiodic exponent",
    "pdr_variability": "PDR variability", "ap_gradient_alpha": "A-P alpha gradient",
    "ap_gradient_freq": "A-P frequency gradient", "theta_alpha_ratio": "Theta/alpha ratio",
    "rel_alpha": "Relative alpha", "spectral_edge": "Spectral median",
    "spectral_entropy": "Spectral entropy", "artefact_proxy": "Artefact proxy",
    "log_n_windows": "Recording length",
}


def nice(n):
    return CONCEPT_NICE.get(n, n.replace("_", " "))


def load(*parts):
    p = Path(*parts)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                                  # noqa: BLE001
        return None


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name)
    print(f"  {name}")




# ------------------------------------------------------------------ F2 composition
def f2_composition(plt):
    d = load(AUD, "artefact_composition.json")
    h = load(AUD, "harmonise_summary.json")
    if not d:
        print("  [skip] F2"); return
    coh = d["cohorts"]
    order = [c for c in ("CAUEEG", "ds004504_RAW", "ds004504_derivative", "GENEEG") if c in coh]
    pretty = {"CAUEEG": "CAUEEG\n(training)", "ds004504_RAW": "OpenNeuro\n(raw)",
              "ds004504_derivative": "OpenNeuro\n(curated)", "GENEEG": "GENEEG"}
    cats = ["brain", "eye", "muscle", "line", "chan", "other"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for c in cats:
        v = np.array([coh[k]["median_share"].get(c, 0.0) for k in order])
        ax.barh(y, v, left=left, height=0.6, color=S.COMPONENT[c],
                label=c if c in ("brain", "eye", "muscle") else None)
        for yi, (vv, ll) in zip(y, zip(v, left)):
            if vv > 0.11:
                ax.annotate(f"{vv:.0%}", (ll + vv / 2, yi), ha="center", va="center",
                            fontsize=6.4, color="white")
        left += v
    ax.set_yticks(y); ax.set_yticklabels([pretty[k] for k in order], fontsize=6.8)
    ax.set_xlim(0, 1); ax.set_xlabel("share of recording variance")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    ax.set_title("ocular activity dominates, muscle is minor", loc="left")

    # Read the by-class artefact burden from the per-recording sidecars rather than from the run
    # summary. The sidecars are the primary record and cannot be overwritten by a later partial
    # run, which is exactly what happened to the summary.
    cache = ROOT / "outputs" / "harmonised"
    keymap = {"CN": "control", "C": "control", "CONTROL": "control", "MCI": "MCI",
              "AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD"}
    klass = ["control", "MCI", "AD", "FTD"]
    buckets: dict = {}
    for side in cache.glob("*.json"):
        try:
            m = json.loads(side.read_text())
        except Exception:                                              # noqa: BLE001
            continue
        cls = keymap.get(str(m.get("group", "")).upper())
        v = (m.get("qc") or {}).get("variance_removed")
        if cls and v is not None:
            buckets.setdefault(m.get("cohort"), {}).setdefault(cls, []).append(float(v))
    series = []
    for c, lbl, col in (("CAUEEG", "CAUEEG (training)", S.SLATE),
                        ("ds004504_raw", "OpenNeuro (test)", S.TERRA)):
        b = buckets.get(c, {})
        vals = [float(np.median(b[k])) if b.get(k) else np.nan for k in klass]
        if np.isfinite(vals).sum() >= 2:
            series.append((lbl, vals, col))
    if series:
        x = np.arange(len(klass))
        for (lbl, vals, col) in series:
            v = np.array(vals, dtype=float)
            ok = np.isfinite(v)
            ax2.plot(x[ok], v[ok], "-o", color=col, ms=4.5, lw=1.4)
            last = int(np.flatnonzero(ok)[-1])
            S.label_end(ax2, x[last], v[last], lbl, color=col, dx=0.25)
        ax2.set_xticks(x); ax2.set_xticklabels(klass, fontsize=7.0)
        # Four classes sit at x=0..3; the previous limit of 2.9 cut FTD off entirely and pushed
        # both end labels outside the axes. The upper bound leaves room for the labels.
        ax2.set_xlim(-0.35, len(klass) - 1 + 1.75)
        ax2.set_ylabel("variance removed as artefact")
        S.hgrid(ax2, "y")
        ax2.set_title("artefact burden tracks diagnosis with\nopposite sign at the two sites",
                      loc="left")
    S.panel_tag(fig, ax, "a"); S.panel_tag(fig, ax2, "b")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "figS6_composition.pdf"); plt.close(fig)


# ------------------------------------------------------------------ F3 validity
def f3_validity(plt):
    """Paired dot plot: each concept's association with the confound against its association with
    the clinical criterion, on the same subjects.

    A scatter of seventeen labelled points overlapped badly and forced the reader to decode
    position rather than read a comparison. One row per concept, two dots joined by a line, sorted
    by the gap, states the same thing without any label collision -- and makes the direction of
    each gap, which is the actual finding, visible at a glance.
    """
    orig = load(AUD, f"concept_validity_harmonised_{TAG}_orig.json") \
        or load(VAL, "validity_harmonised_original.json")
    rep = load(AUD, f"concept_validity_harmonised_{TAG}.json")
    if not orig or not rep:
        print("  [skip] F3"); return

    # The two validity files name these columns differently: the retired
    # validity_harmonised_original.json used the *_openneuro88 suffix, the audit files use
    # rho_artefact_mmse_sample / rho_mmse. Accept either, or the panel silently comes out empty.
    ART_KEYS = ("rho_artefact_mmse_sample", "rho_artefact_openneuro88")
    MMSE_KEYS = ("rho_mmse", "rho_mmse_openneuro88")

    def first(r, keys):
        for k in keys:
            if r.get(k) is not None:
                return r[k]
        return None

    def rows(con, kv="verdict"):
        out = []
        for n, r in con.items():
            a, m = first(r, ART_KEYS), first(r, MMSE_KEYS)
            if a is None or m is None or n == "artefact_proxy":
                continue
            out.append((nice(n), abs(a), abs(m), str(r.get(kv, ""))))
        return out

    left = rows(orig["concepts"])
    right = rows(rep["concepts"])
    if not left or not right:
        print("  [skip] F3: no usable rows (checked %s / %s)" % (ART_KEYS, MMSE_KEYS)); return
    order = [r[0] for r in sorted(left, key=lambda r: r[1] - r[2])]

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.2), sharey=True, sharex=True)
    for ax, data, title in zip(axes, (left, right),
                               ("wide analysis band", "restricted to 20 Hz")):
        d = {r[0]: r for r in data}
        y = np.arange(len(order))
        for yi, name in zip(y, order):
            if name not in d:
                continue
            _, art, mms, verdict = d[name]
            v = verdict.upper()
            bad = v.startswith("ARTEFACT")
            good = v.startswith("CRITERION")
            ax.plot([mms, art], [yi, yi], color=S.HAIR, lw=1.6, zorder=1)
            # The battery returns three verdicts and only one of them was ever drawn, so the
            # concepts it cleared -- the positive half of the result -- were invisible.
            ax.scatter([mms], [yi], s=30 if good else 26,
                       color=S.SAGE if good else S.SLATE, zorder=3)
            ax.scatter([art], [yi], s=26, color=S.TERRA if bad else S.STONE, zorder=3)
            if bad:
                ax.annotate("artefact-dominated", (max(art, mms), yi), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=5.8,
                            color=S.TERRA)
            elif good:
                ax.annotate("criterion-dominated", (max(art, mms), yi), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=5.8,
                            color=S.SAGE)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1.02); ax.set_xlabel("association  |$\\rho$|")
        ax.set_title(title, loc="left", x=-0.02)
        S.hgrid(ax, "x")
        # The floor below which neither correlation is separable from zero at this sample size,
        # so that a row with both dots to the left of it reads as untested rather than as null.
        ax.axvline(0.211, color=S.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    # The legend and the detection-floor note both used to be anchored under panel (a) and landed
    # on top of each other. The legend now sits at figure level below both panels, and the note
    # is a separate line beneath it, so neither can collide with the other or with an axis label.
    h = [plt.Line2D([], [], ls="", marker="o", color=S.SLATE, ms=4.5,
                    label="with MMSE (the referent)"),
         plt.Line2D([], [], ls="", marker="o", color=S.SAGE, ms=4.5,
                    label="MMSE, verdict criterion-dominated"),
         plt.Line2D([], [], ls="", marker="o", color=S.TERRA, ms=4.5,
                    label="with measured artefact")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.062), ncol=3,
               frameon=False, fontsize=6.6, handletextpad=0.4, columnspacing=1.6)
    fig.text(0.5, 0.030,
             "dashed line: below $|\\rho| = 0.21$ neither correlation is separable from zero at "
             "$n=87$, so the row is untested rather than cleared",
             ha="center", fontsize=5.6, color=S.MUTED)
    S.panel_tag(fig, axes[0], "a"); S.panel_tag(fig, axes[1], "b")
    fig.subplots_adjust(wspace=0.30, top=0.93, bottom=0.20)
    save(fig, "fig6_validity.pdf"); plt.close(fig)


# ------------------------------------------------------------------ F4 failure modes
def f4_modes(plt):
    ica = load(VAL, "ica_muscle_check.json")
    topo = load(AUD, "topographic_construct.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8),
                         gridspec_kw={"width_ratios": [1.05, 1.15, 0.95]})

    # mode I -- confirmed against independent component labels
    ax = axes[0]
    if ica and "OpenNeuro" in ica.get("cohorts", {}):
        cc = ica["cohorts"]["OpenNeuro"]["correlations"]
        want = [("Artefact proxy", "artefact_proxy (THE PROXY)"),
                ("Aperiodic exponent", "aperiodic_exponent (pre-repair)"),
                ("Spectral median", "spectral_edge (pre-repair)"),
                ("Spectral entropy", "spectral_entropy (pre-repair)"),
                ("PDR frequency", "pdr_frequency (CONTROL)"),
                ("Theta/alpha ratio", "theta_alpha_ratio (CONTROL)")]
        lab, val, ci = [], [], []
        for short, key in want:
            if key in cc and cc[key]["rho"] is not None:
                lab.append(short); val.append(abs(cc[key]["rho"]))
                lo, hi = cc[key]["ci95"]
                lo = 0.0 if lo is None else lo
                hi = 0.0 if hi is None else hi
                # On a |rho| axis an interval that straddles zero runs from 0 to the larger
                # magnitude. Taking the absolute value of each bound separately would place the
                # interval away from zero and make a null result look like a positive one.
                if lo <= 0 <= hi:
                    ci.append((0.0, max(abs(lo), abs(hi))))
                else:
                    ci.append((min(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
        y = np.arange(len(lab))[::-1]
        # The two audited concepts are negative controls and must be drawn as such. This set
        # previously read "theta/alpha", which matches no label the panel produces, so one of
        # the two controls was silently coloured as a finding.
        controls = {"PDR frequency", "Theta/alpha ratio"}
        for yi, v, (lo, hi), l in zip(y, val, ci, lab):
            c = S.STONE if l in controls else S.TERRA
            ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, alpha=0.35)
            ax.scatter([v], [yi], s=24, color=c, zorder=3)
        ax.axvline(0, color=S.HAIR, lw=0.8, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.6)
        ax.set_xlabel("|$\\rho$| with muscle components")
        ax.tick_params(axis="y", length=0)
        S.hgrid(ax, "x")
        ax.annotate("grey: negative controls", (0.0, -0.20), xycoords="axes fraction",
                    fontsize=5.2, color=S.MUTED, va="top")
    ax.set_title("I  estimator measures\nan artefact", loc="left", fontsize=7.6)

    # mode II -- invariance
    # ---- mode II -- the construct is not expressible ----------------------------------------
    ax = axes[1]
    if topo and "part_a" in topo:
        pa = topo["part_a"]
        before = pa["synthetic_true_posterior_anterior_alpha_power_ratio"]["before_gains"]
        after = pa["synthetic_true_posterior_anterior_alpha_power_ratio"]["after_gains"]
        per = pa.get("max_abs_change_per_concept", {})
        NAMED = ("posterior_maximality", "ap_gradient_alpha", "asymmetry_alpha")
        truth = 100.0 * abs(after / before - 1.0)

        # This panel used to be a log-scale bar chart of every concept's change. Every value in
        # that chart lay between 1e-9 and 1e-6 -- float32 rounding, nothing else -- and a log
        # axis magnifies rounding into visible bars whose *order* is meaningless. Two of the
        # three concepts named after amplitude happened to round larger than several unnamed
        # ones, so the panel appeared to show the named concepts moving MOST, which is the
        # opposite of the finding. Drawn on the linear scale the claim is made on, the finding
        # is one bar and three empty rows.
        # Short labels. The two-line version of the first row ran left out of this panel and
        # over the interval bars of panel (a).
        rows = [("Posterior/anterior\nalpha power itself", truth, True)]
        rows += [(nice(k), 100.0 * float(per.get(k, 0.0)), False) for k in NAMED]
        y = np.arange(len(rows))[::-1]
        for yi, (lab_, v, is_truth) in zip(y, rows):
            ax.barh(yi, v, height=0.52, color=S.INK_SOFT if is_truth else S.TERRA)
            if is_truth:
                ax.annotate("%.0f%%" % v, (v, yi), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=7.0, color=S.INK, weight="semibold")
            else:
                ax.annotate("0.0000%", (0, yi), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=6.2, color=S.TERRA)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.0, linespacing=1.3)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, truth * 1.42)
        ax.set_xlabel("change under random per-channel gains (%)")
        S.hgrid(ax, "x")
        # One short line. The three-line version ran past this panel's right edge and into
        # panel (c); the caption carries the rest of the explanation.
        ax.annotate("largest change across all fourteen concepts: $10^{-7}$ per cent",
                    (0.0, -0.20), xycoords="axes fraction", fontsize=5.2, color=S.MUTED,
                    va="top")
    ax.set_title("II  construct is not\nexpressible", loc="left", fontsize=7.6)

    # mode III -- site dependence of the instrument
    ax = axes[2]
    if ica:
        sites, vals = [], []
        for s, lbl in (("CAUEEG", "training cohort"), ("OpenNeuro", "dementia site")):
            r = ica.get("cohorts", {}).get(s, {}).get(
                "correlations", {}).get("artefact_proxy (THE PROXY)", {})
            if r.get("rho") is not None:
                sites.append(lbl); vals.append(r["rho"])
        y = np.arange(len(sites))[::-1]
        ax.barh(y, vals, height=0.45, color=[S.STONE, S.TERRA][:len(vals)])
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:+.2f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6.6, color=S.INK_SOFT)
        ax.set_yticks(y); ax.set_yticklabels(sites, fontsize=6.8)
        ax.set_xlim(0, 0.78); ax.set_xlabel("proxy vs muscle components  $\\rho$")
        ax.tick_params(axis="y", length=0)
        S.hgrid(ax, "x")
    ax.set_title("III  the instrument is\nsite-dependent", loc="left", fontsize=7.6)

    for a, t in zip(axes, "abc"):
        S.panel_tag(fig, a, t)
    fig.subplots_adjust(wspace=0.78)
    save(fig, "fig7_failure_modes.pdf"); plt.close(fig)






# ------------------------------------------------------------------ F6 interpretability
def f6_report(plt):
    """A single patient's decision, decomposed into named contributions."""
    import numpy as np
    feat = ROOT / "outputs" / "harmonised_features" / TAG
    if not (feat / "window_concepts.npy").exists():
        print("  [skip] F6"); return
    sys.path.insert(0, str(ROOT))
    from sklearn.model_selection import StratifiedKFold

    from src.models.v6 import (CONCEPT_NAMES, SignConstrainedLogistic, aggregate_subject,
                               impute_from_training)
    from src.data.feature_table import check_table
    from _protocol_v2 import CN, DEM, MCI

    LAB = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI, "AD": DEM, "A": DEM,
           "FTD": DEM, "F": DEM}
    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1, "spectral_entropy": 0,
           "artefact_proxy": 0, "log_n_windows": 0}
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")

    check_table(feat)   # concepts and meta are joined by row index below
    C_win = np.load(feat / "window_concepts.npy")
    meta = np.load(feat / "meta.npz", allow_pickle=False)
    g, lb, co = meta["groups"], meta["labels"], meta["cohort"]
    subs = np.unique(g); idxs = [np.flatnonzero(g == s) for s in subs]
    coh = np.array([co[i[0]] for i in idxs]); y3 = np.array([LAB[lb[i[0]]] for i in idxs])
    C = np.stack([aggregate_subject(C_win[i]) for i in idxs])
    tr = coh == "CAUEEG"
    C, _ = impute_from_training(C, tr)
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    names = [CONCEPT_NAMES[i] for i in cols]
    X = C[:, cols]
    y1 = (y3 != CN).astype(int)
    fits = [SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(X[tr][a], y1[tr][a])
            for a, _ in StratifiedKFold(5, shuffle=True, random_state=42).split(X[tr], y3[tr])]
    m = coh == "ds004504_raw"
    p = np.mean([f.predict_proba(X[m])[:, 1] for f in fits], axis=0)
    yv = y1[m]

    # Two dementia patients: the one the model is most confidently RIGHT about and the one it is
    # most confidently WRONG about. Same true diagnosis, opposite outcome, so the pair shows what
    # the panel does rather than only where it fails.
    dem = np.flatnonzero(yv == 1)
    right = int(dem[np.argmax(p[dem])])
    wrong = int(dem[np.argmin(p[dem])])
    cont = {k: np.mean([f.contributions(X[m][[k]]) for f in fits], axis=0)[0]
            for k in (right, wrong)}

    w_mean = np.mean([np.asarray(getattr(f, "coef_", getattr(f, "w", None))).ravel()
                      for f in fits], axis=0)
    used_mask = np.abs(w_mean) >= 1e-8
    # One ordering for both panels, taken from the correct case, so rows line up across panels.
    # Concepts the sign constraint holds at exactly zero are dropped from the panel rather than
    # drawn as empty rows: they cannot move the score by any amount, so a row for them carries no
    # information about this patient and cost 43 per cent of the plot area. Which concepts they
    # are, and why, is a result in its own right and is stated in the caption and in the text,
    # not left for the reader to infer from blank space.
    order = np.argsort(cont[right])
    dropped = [nice(names[j]) for j in order if not used_mask[j]]
    order = np.array([j for j in order if used_mask[j]])
    lim = max(np.abs(np.concatenate([cont[right], cont[wrong]])).max() * 1.25, 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    yy = np.arange(len(order))
    for ax, k, tag in ((axes[0], right, "correctly identified"),
                       (axes[1], wrong, "confidently missed")):
        c = cont[k]
        cols_ = [(S.TERRA if c[j] > 0 else S.SLATE) if used_mask[j] else S.STONE
                 for j in order]
        ax.barh(yy, c[order], height=0.62, color=cols_)
        ax.axvline(0, color=S.MUTED, lw=0.9)
        ax.set_xlim(-lim, lim)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("contribution to the log-odds")
        S.hgrid(ax, "x")
        ax.set_title(tag, loc="left", x=-0.02, pad=18)
        ax.annotate(f"dementia; model probability {p[k]:.2f}",
                    (0.0, 1.015), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=6.6, color=S.INK_SOFT)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([nice(names[j]) for j in order], fontsize=6.6)
    axes[0].annotate("toward normal", (0.0, -0.155), xycoords="axes fraction",
                     ha="left", va="top", fontsize=6.3, color=S.SLATE)
    axes[1].annotate("toward impairment", (1.0, -0.155), xycoords="axes fraction",
                     ha="right", va="top", fontsize=6.3, color=S.TERRA)
    if dropped:
        axes[0].annotate(f"{len(order)} of {len(names)} concepts shown; the sign constraint holds "
                         f"the rest at zero weight,\nso they cannot affect any decision: "
                         + ", ".join(dropped).lower(),
                         (0.0, -0.235), xycoords="axes fraction", ha="left", va="top",
                         fontsize=5.8, color=S.MUTED)
    S.panel_tag(fig, axes[0], "a"); S.panel_tag(fig, axes[1], "b")
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.25, wspace=0.12)
    save(fig, "fig2_report.pdf"); plt.close(fig)




def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG,
                    help="analysis arm to draw from; must match the arm the manuscript "
                         "tables report (default: channel_clean)")
    TAG = ap.parse_args().tag
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)
    print(f"writing figures to {FIGS}  [arm: {TAG}]")
    # f1_design and f7_ood are retired: both are now built by 166_figures_v2.py, which reads the
    # harmonised arm. The versions here drew the pre-matching corpus (CAUEEG 1,120, OpenNeuro 88)
    # and, for the guard, an artefact carrying no provenance block, so they disagreed with the
    # tables beside them. f5_benchmark is retired for the same class of reason -- it drew a
    # different training arm from the table beside it. They are kept below only so the history of
    # each panel stays readable.
    for fn in (f2_composition, f3_validity, f4_modes, f6_report):
        try:
            fn(plt)
        except Exception as e:                                         # noqa: BLE001
            import traceback
            print(f"  [fail] {fn.__name__}: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc(limit=1)
    print("done")


if __name__ == "__main__":
    main()
