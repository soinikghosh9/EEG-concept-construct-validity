"""Rebuilt and new manuscript figures.

Four panels are produced here rather than in `144`, either because the original drifted from its
artefact or because a central claim of the paper had no figure at all.

    F1  study design: the funnel from candidate cohorts to analysed corpus to tasks.
        REBUILT. The previous version was two bar charts that restated table 1, and its counts
        (CAUEEG 1,120, OpenNeuro 88) were the pre-matching ones, so it disagreed with every table
        in the paper.
    F6  task-by-model landscape (fig6_landscape). NEW. The claim that the question moves discrimination further
        than the model had no figure; a fourteen-by-four table is not a picture.
    F8  concept validity against concept use (fig8_concepts). NEW. The sharpest result in the paper -- that the
        concepts the battery can adjudicate are not the ones the model uses -- had no figure.
    F7  the operating point. REBUILT. The previous version read `covariate_shift.json`, which
        carries no provenance block, was computed on the pre-matching corpus, and included two
        cohorts the study excludes; it therefore printed 24 per cent for the dementia site where
        the text says 13.8. Panel b previously plotted a hard-coded 0.582.

    python experiments/166_figures_v2.py
Out: jne_build/Figs/fig1_design.pdf, fig6_landscape.pdf, fig8_concepts.pdf, fig7_operating.pdf
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
FIGS = ROOT / "jne_build" / "Figs"

TAG = "channel_clean"
TASKS = [("CN_vs_AD", "Alzheimer's"), ("CN_vs_dementia", "Dementia"),
         ("CN_vs_impaired", "Screening"), ("CN_vs_MCI", "Prodromal")]

CLASSICAL = {"random_forest", "gradient_boosting", "SVM", "logistic"}

CONCEPT_NICE = {
    "rel_theta": "Relative theta", "pdr_frequency": "PDR frequency",
    "posterior_maximality": "Posterior maximality", "asymmetry_alpha": "Alpha asymmetry",
    "asymmetry_freq": "Frequency asymmetry", "pdr_prominence": "PDR prominence",
    "slowing_ratio": "Slowing ratio", "aperiodic_exponent": "Aperiodic exponent",
    "pdr_variability": "PDR variability", "ap_gradient_alpha": "A-P alpha gradient",
    "ap_gradient_freq": "A-P frequency gradient", "theta_alpha_ratio": "Theta/alpha ratio",
    "rel_alpha": "Relative alpha", "spectral_edge": "Spectral median"}


def load(name):
    p = AUD / name
    if not p.exists():
        raise SystemExit("missing artefact: %s" % p)
    return json.loads(p.read_text())


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name)
    print("  %s" % name)


# ------------------------------------------------------------------ F1 concept-to-decision
def f1_design(plt):
    """The instrument, and where each question in this paper attaches to it.

    Matplotlib text is not LaTeX; no escapes here. Box extents stay inside the axis limits with a
    margin for the rounded border, or the outermost box is clipped at the figure edge.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(-2, 102); ax.set_ylim(0, 54); ax.axis("off")

    def box(x, y, w, h, title, body, ec, fc="#FFFFFF"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.1",
                                    linewidth=1.0, facecolor=fc, edgecolor=ec, zorder=2))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center", fontsize=8.2,
                color=S.INK, weight="semibold", zorder=3, linespacing=1.3)
        if body:
            ax.text(x + w / 2, y + (h - 7.0) / 2 + 0.6, body, ha="center", va="center",
                    fontsize=6.9, color=S.INK_SOFT, zorder=3, linespacing=1.5)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.0, color=S.MUTED, zorder=1,
                                     shrinkA=1, shrinkB=1))

    TOP, H = 25.0, 24.0
    MID = TOP + H / 2
    # x, width for the four stages; the last ends at 96 so its border stays inside the axes
    X = [(0, 19), (25, 27), (57, 20), (81, 15)]

    box(X[0][0], TOP, X[0][1], H, "Resting EEG",
        "19 electrodes\n96 s of gradable\nrecording per subject\nidentical for everyone",
        S.HAIR, "#F4F3EF")
    box(X[1][0], TOP, X[1][1], H, "15 named clinical concepts", "", S.SLATE, "#EEF2F8")
    box(X[2][0], TOP, X[2][1], H, "Sign-constrained\nlinear head",
        "each concept enters with\nthe direction clinical\nknowledge specifies;\n"
        "contributions sum to\nthe score exactly",
        S.SLATE, "#EEF2F8")
    box(X[3][0], TOP, X[3][1], H, "Referral\ndecision",
        "control vs\nimpaired,\nthen MCI vs\ndementia",
        S.SAGE, "#F1F5EF")
    for (x, w), (nx, _) in zip(X[:-1], X[1:]):
        arrow(x + w + 0.8, nx - 0.8, MID)

    # the concept list, left-aligned so the four category labels form a column
    rows = [("rhythm", "PDR frequency, prominence,\nvariability"),
            ("slowing", "relative theta, theta/alpha,\nslowing ratio"),
            ("topography", "posterior maximality,\nA-P gradient, asymmetry"),
            ("background", "aperiodic exponent, edge")]
    ry = TOP + H - 7.6
    for lab, items in rows:
        ax.text(X[1][0] + 1.6, ry, lab, ha="left", va="top", fontsize=6.4, color=S.SLATE,
                weight="semibold", zorder=3)
        ax.text(X[1][0] + 9.4, ry, items, ha="left", va="top", fontsize=6.4, color=S.INK_SOFT,
                zorder=3, linespacing=1.4)
        ry -= 4.3 if "\n" in items else 2.6

    # where each question in the paper attaches
    QY = 14.0
    questions = [
        (X[1], "Do they measure what\nthey are named?",
         "seven checks, four of which\ncan reject a concept"),
        (X[2], "Which concepts carry\nthe decision?",
         "permutation importance\non the blind site"),
        (X[3], "Where is the threshold,\nand does it move?",
         "set from training prevalence,\nno target labels"),
    ]
    for (x, w), q, how in questions:
        cx = x + w / 2
        ax.add_patch(FancyArrowPatch((cx, TOP - 0.8), (cx, QY + 6.6), arrowstyle="-",
                                     linewidth=0.8, color=S.TERRA, linestyle=(0, (2, 2)),
                                     zorder=1))
        ax.text(cx, QY + 4.4, q, ha="center", va="center", fontsize=6.9, color=S.TERRA,
                weight="semibold", linespacing=1.35)
        ax.text(cx, QY - 0.6, how, ha="center", va="center", fontsize=6.2, color=S.MUTED,
                linespacing=1.4)

    ax.text(0, 6.0, "Every model is trained on one cohort and evaluated at clinics that "
                    "contributed nothing to it. Artefact burden and recording\nlength are "
                    "computed and reported but never enter the decision.",
            fontsize=6.4, color=S.INK_SOFT, va="top", linespacing=1.5)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "fig1_design.pdf"); plt.close(fig)




# ------------------------------------------------------------------ F6 validity against use
def f8_concepts(plt):
    prof = load("concept_task_profile.json")["tasks"]
    names = list(prof["CN_vs_AD"]["concepts"])

    imp = np.array([[prof[t]["concepts"][n]["importance"] for t, _ in TASKS] for n in names])
    sig = np.array([[prof[t]["concepts"][n]["significant"] for t, _ in TASKS] for n in names])
    verdict = [prof["CN_vs_AD"]["concepts"][n]["verdict"] for n in names]
    rho = np.array([abs(prof["CN_vs_AD"]["concepts"][n]["rho_mmse"] or 0) for n in names])

    order = np.argsort(-imp.max(axis=1))
    names = [names[i] for i in order]
    imp, sig, rho = imp[order], sig[order], np.asarray(rho)[order]
    verdict = [verdict[i] for i in order]

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- (a) per-task importance, one row per concept
    from matplotlib.patches import Rectangle

    M = np.clip(imp, -0.02, 0.15)
    im = ax.imshow(M, cmap=S.sequential(plt), vmin=-0.02, vmax=0.15, aspect="auto")
    # Significance is drawn as a ring round the cell rather than a glyph: no font dependency, and
    # it survives greyscale printing, which a coloured asterisk does not.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if sig[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=S.TERRA, linewidth=1.4, zorder=4))
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([lab for _, lab in TASKS], fontsize=6.2, rotation=28, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([CONCEPT_NICE.get(n, n) for n in names], fontsize=6.1)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("what each concept contributes,\nby clinical question", loc="left", pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.02)
    cb.ax.tick_params(labelsize=5.6, length=0); cb.outline.set_visible(False)
    ax.annotate("outlined cell: interval excludes zero", (0.0, -0.235),
                xycoords="axes fraction", fontsize=5.5, color=S.MUTED, va="top")

    # ---- (b) validity against use, as a slope chart
    # A scatter puts nine of the fourteen concepts on the same point at the origin. Ranking each
    # concept on both criteria and joining them makes the disjunction the thing the eye sees:
    # if validity predicted use the lines would run flat, and they do not.
    VC = {"criterion-dominated": S.SAGE, "artefact-dominated": S.TERRA, "unresolvable": S.STONE}
    best = imp.max(axis=1)
    k = len(names)
    left = np.argsort(np.argsort(-rho))          # 0 = strongest criterion correlation
    right = np.argsort(np.argsort(-best))        # 0 = largest contribution

    for i, n in enumerate(names):
        c = VC.get(verdict[i], S.STONE)
        lw = 1.5 if best[i] > 0.01 else 0.7
        al = 0.95 if best[i] > 0.01 else 0.45
        ax2.plot([0, 1], [left[i], right[i]], color=c, lw=lw, alpha=al, zorder=2,
                 solid_capstyle="round")
        ax2.scatter([0, 1], [left[i], right[i]], s=13, color=c, zorder=3, linewidths=0)
        ax2.annotate(CONCEPT_NICE.get(n, n), (0, left[i]), xytext=(-6, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=5.4,
                     color=S.INK if best[i] > 0.01 else S.MUTED)
        lab = ("%.3f" % best[i]) if best[i] > 1e-9 else "none"
        ax2.annotate(lab, (1, right[i]), xytext=(6, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=5.4,
                     color=S.INK if best[i] > 0.01 else S.MUTED)

    ax2.set_xlim(-0.52, 1.42); ax2.set_ylim(k - 0.4, -1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["ranked by\ncriterion validity", "ranked by\ncontribution"], fontsize=5.9)
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("the concepts that can be validated\nare not the ones that get used",
                  loc="left", pad=8)
    # Build the key from the verdicts this panel actually contains. It previously listed all
    # three, including artefact-dominated, which cannot appear here: the only artefact-dominated
    # quantity in the reported arm is spectral entropy, and spectral entropy is the one concept
    # excluded from the classifier, so it is not among the fourteen plotted. A key entry with no
    # mark on the panel reads as a mark the reader has failed to find.
    present = [v for v in VC if v in set(verdict)]
    handles = [plt.Line2D([], [], marker="o", ls="-", ms=3.6, lw=1.4, color=VC[v], label=v)
               for v in present]
    ax2.legend(handles=handles, fontsize=5.3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.42, -0.235), ncol=len(handles), handletextpad=0.35,
               columnspacing=1.0)
    missing = [v for v in VC if v not in set(verdict)]
    if missing:
        ax2.annotate("no concept here is %s: the one that is, spectral entropy,\n"
                     "is the concept excluded from the classifier" % missing[0],
                     (0.42, -0.30), xycoords="axes fraction", ha="center", va="top",
                     fontsize=5.0, color=S.MUTED, linespacing=1.5)

    S.panel_tag(fig, ax, "a"); S.panel_tag(fig, ax2, "b")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.85, bottom=0.19)
    save(fig, "fig8_concepts.pdf"); plt.close(fig)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)

    figs = {"1": f1_design, "8": f8_concepts}
    for k, fn in figs.items():
        if args.only and k not in args.only:
            continue
        fn(plt)


if __name__ == "__main__":
    main()
