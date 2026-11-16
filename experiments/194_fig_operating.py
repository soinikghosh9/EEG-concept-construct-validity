"""Rebuild the operating-point figure (printed figure 9, file fig9_operating.pdf).

Drawn in receiver-operating characteristic space, which no other figure in this paper uses and
which states the subsection's claim without needing a caption: the curve is the classifier and
does not move, the dot is the decision and moves a long way.

Panel (a) plots where the eleven comparators' decisions land at the label-free threshold, one
dot per model per task. Their tight per-task clusters are the point: the models agree closely
with each other and all sit in the same wrong place, high and to the left, so what puts them
there is the rule rather than any model's judgement. Panel (b) takes the reported instrument's
curve at the dementia site and marks the three threshold rules on it.

Panel (a) shows the comparators and panel (b) the concept model, because that is how the
subsection reports them: its first paragraph quotes medians over the eleven and its second
quotes the concept model against an oracle. Those are different quantities and the caption says
so; the concept model at the same rule is more balanced than the comparators, not less.

Earlier versions of this figure pair were a four-point line with a refusal bar chart, then two
crossing series, then a paired-dot chart. The last repeated the grammar of the construct-validity
figure, which already joins two dots per row, so the two read as the same exhibit.

    python experiments/194_fig_operating.py
Out: jne_build_revised/Figs/fig9_operating.pdf
"""
from __future__ import annotations

import json
import statistics as stx
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import _figstyle_2026 as S  # noqa: E402

FIGS = ROOT / "jne_build_revised" / "Figs"
EXCLUDE: set = set()
AUD = ROOT / "outputs" / "results" / "audit"
PRED = AUD / "predictions_per_task"
TAG = "channel_clean"
SITE = "OpenNeuro|CN_vs_dementia"

#: task, label, target/source prevalence ratio, where to put the cluster label
TASKS = [("CN_vs_impaired", "Screening", 1.14, (13, 7)),
         ("CN_vs_AD", "Alzheimer's", 1.69, (8, 15)),
         ("CN_vs_dementia", "Dementia", 1.97, (8, -16)),
         ("CN_vs_MCI", "Prodromal", 0.94, (13, -9))]
NEAR_CHANCE = "CN_vs_MCI"

RULES = [("target", "target prevalence", (13, -8), "left"),
         ("source", "label-free rule", (13, -8), "left"),
         ("youden", "oracle (needs labels)", (13, -8), "left")]


def roc(y, s):
    o = np.argsort(-np.asarray(s))
    y = np.asarray(y)[o]
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    return np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)

    per = json.load(open(AUD / "deep_per_task.json"))["tasks"]
    op = json.load(open(AUD / f"threshold_sensitivity_{TAG}.json")
                   )["sites"][SITE]["models"]["concept_bottleneck"]["operating"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    COL = {"CN_vs_impaired": S.SLATE, "CN_vs_AD": S.SAGE,
           "CN_vs_dementia": S.PLUM, "CN_vs_MCI": S.STONE}

    # ---------------------------------------------------------------- (a) where decisions land
    for task, lab, ratio, off in TASKS:
        M = [m for k, m in per[task]["models"].items()
             if "at_source_prevalence" in m and k not in EXCLUDE]
        se = [m["at_source_prevalence"]["sensitivity"] for m in M]
        sp = [m["at_source_prevalence"]["specificity"] for m in M]
        live = task != NEAR_CHANCE
        c = COL[task]
        axa.scatter([1 - v for v in sp], se, s=13, color=c, alpha=0.55, zorder=3,
                    linewidths=0)
        mx, my = 1 - stx.median(sp), stx.median(se)
        axa.scatter([mx], [my], s=64, color=c, zorder=5, linewidths=1.3,
                    edgecolors=S.PAPER)
        axa.annotate(f"{lab}  ×{ratio:.2f}  {100*my:.0f}% found",
                     (mx, my), xytext=off, textcoords="offset points",
                     ha="left", va="center", fontsize=5.8,
                     color=S.INK_SOFT if live else S.MUTED)
    axa.set_title("where the label-free decision lands", loc="left", pad=8)
    axa.annotate("eleven models, one dot each, large dot the median. The clusters are tight,\n"
                 "so what puts a decision in the top-left corner is the rule, not the model.",
                 (0.0, -0.26), xycoords="axes fraction", fontsize=5.4, color=S.MUTED,
                 va="top", linespacing=1.4)

    # ---------------------------------------------------------------- (b) one curve, three rules
    # The curve must be the instrument whose operating points are marked: the two-stage concept
    # model's stored posteriors at the dementia site, scored as in 155_threshold_sensitivity.py.
    from _protocol_v2 import CN, pairwise_score, safe_auc, target_subset
    d = np.load(AUD / "predictions" / TAG / "concept_bottleneck__ds004504_raw.npz")
    mt, _ = target_subset(d["y3"], "CN_vs_dementia")
    yb = (d["y3"][mt] != CN).astype(int)
    sb = pairwise_score(d["posterior"][mt], "CN_vs_dementia")
    assert abs(safe_auc(yb, sb) - json.load(open(AUD / f"threshold_sensitivity_{TAG}.json"))
               ["sites"][SITE]["models"]["concept_bottleneck"]["auc"]) < 1e-9
    fpr, tpr = roc(yb, sb)
    axb.plot(fpr, tpr, color=S.PLUM, lw=1.9, zorder=3)
    axb.fill_between(fpr, 0, tpr, color=S.PLUM, alpha=0.06, zorder=1)
    for key, lab, off, ha in RULES:
        o = op[key]
        px, py = 1 - o["specificity"], o["sensitivity"]
        axb.scatter([px], [py], s=54, color=S.TERRA, zorder=6, linewidths=1.2,
                    edgecolors=S.PAPER)
        axb.annotate(f"{lab}  {100*py:.0f}%", (px, py), xytext=off,
                     textcoords="offset points", ha=ha, va="center", fontsize=5.8,
                     color=S.INK_SOFT)
    axb.set_title("three thresholds, one curve", loc="left", pad=8)
    axb.annotate("the reported instrument at the dementia site. Moving the threshold slides the\n"
                 "decision along a curve that does not change: what fails is the dot, not the line.",
                 (0.0, -0.26), xycoords="axes fraction", fontsize=5.4, color=S.MUTED,
                 va="top", linespacing=1.4)

    for ax in (axa, axb):
        ax.plot([0, 1], [0, 1], color=S.HAIR, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("false positive rate (1 $-$ specificity)")
        ax.set_aspect("equal", adjustable="box")
        S.strip(ax)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_visible(True)
            ax.spines[sp_].set_color(S.HAIR)
    axa.set_ylabel("sensitivity")

    S.panel_tag(fig, axa, "a")
    S.panel_tag(fig, axb, "b")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.24, wspace=0.26)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig9_operating.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for task, lab, ratio, _ in TASKS:
        M = [m for k, m in per[task]["models"].items()
             if "at_source_prevalence" in m and k not in EXCLUDE]
        se = [m["at_source_prevalence"]["sensitivity"] for m in M]
        sp = [m["at_source_prevalence"]["specificity"] for m in M]
        print(f"   (a) {lab:11s} x{ratio:.2f}  median sens {stx.median(se):.3f} "
              f"spec {stx.median(sp):.3f}  ({len(M)} models)")
    for key, lab, _, _ in RULES:
        o = op[key]
        print(f"   (b) {key:8s} sens {o['sensitivity']:.3f} spec {o['specificity']:.3f} "
              f"bal {o['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
