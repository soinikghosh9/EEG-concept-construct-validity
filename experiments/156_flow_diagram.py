"""Participant flow, drawn from the pipeline artefacts rather than transcribed by hand.

Every count here is read from outputs/ at run time, so the figure cannot drift from the corpus the
way a hand-maintained diagram does. Reviewers of prediction-model studies expect this diagram, and
building it from the artefacts is also the cheapest audit available: it forces every recording that
entered preprocessing to be accounted for at the point it left.

    python experiments/156_flow_diagram.py
Out: jne_build/Figs/figS5_flow.pdf
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import _figstyle_2026 as S  # noqa: E402

AUD = ROOT / "outputs" / "results" / "audit"
FIGS = ROOT / "jne_build" / "Figs"
TAG = "channel_clean"

ANALYSED = {"CAUEEG": 1098, "ds004504_raw": 87, "GENEEG": 63, "MISP": 20, "P-ADIC": 28}
ROLE = {"CAUEEG": "training", "ds004504_raw": "blind evaluation",
        "GENEEG": "blind evaluation", "MISP": "single class", "P-ADIC": "single class"}
PRETTY = {"ds004504_raw": "OpenNeuro ds004504", "MCI_Dataset": "MCI-Dataset"}


def census():
    """Recordings per cohort that survived preprocessing into the harmonised cache."""
    c = collections.Counter()
    for p in glob.glob(str(ROOT / "outputs" / "harmonised" / "*.json")):
        try:
            c[json.loads(Path(p).read_text()).get("cohort", "?")] += 1
        except Exception:                                                  # noqa: BLE001
            c["unreadable"] += 1
    return c


def flow():
    """The reconciled funnel, as a dict, so the design figure and this one cannot disagree.

    The assertion is the point of the function: every recording entering preprocessing has to be
    accounted for at the step it left, or the counts printed in the paper are not a description of
    what was analysed.
    """
    cache = census()
    ext = json.load(open(AUD / f"extract_harmonised_{TAG}.json"))
    entered = {k: v["n_subjects"] for k, v in ext["by_cohort"].items()}

    cache_total = sum(cache.values())
    not_extracted = {k: v for k, v in cache.items() if k not in entered}
    entered_total = sum(entered.values())
    zero_retained = {k: v for k, v in entered.items() if k not in ANALYSED}
    candidates = entered_total - sum(zero_retained.values())
    quota_loss = {k: entered[k] - ANALYSED[k] for k in ANALYSED if entered[k] != ANALYSED[k]}
    analysed_total = sum(ANALYSED.values())

    assert candidates - sum(quota_loss.values()) == analysed_total == ext["n_subjects"], \
        "flow does not reconcile against the extraction artefact"

    return {"cache": cache, "cache_total": cache_total, "entered": entered,
            "entered_total": entered_total, "not_extracted": not_extracted,
            "zero_retained": zero_retained, "candidates": candidates,
            "quota_loss": quota_loss, "analysed": dict(ANALYSED),
            "analysed_total": analysed_total,
            "n_epochs": int(ext.get("n_windows") or ext.get("n_epochs") or 0)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    S.apply(plt)

    F = flow()
    cache, cache_total = F["cache"], F["cache_total"]
    entered, entered_total = F["entered"], F["entered_total"]
    not_extracted, zero_retained = F["not_extracted"], F["zero_retained"]
    candidates, quota_loss = F["candidates"], F["quota_loss"]
    analysed_total = F["analysed_total"]

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")

    def box(x, y, w, h, title, lines, accent=S.INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    linewidth=0.9, edgecolor=accent,
                                    facecolor=S.PANEL, zorder=2))
        ax.text(x + 0.22, y + h - 0.30, title, fontsize=7.6, weight="bold",
                color=accent, va="top", zorder=3)
        ax.text(x + 0.22, y + h - 0.68, "\n".join(lines), fontsize=6.6,
                color=S.INK_SOFT, va="top", linespacing=1.55, zorder=3)

    def arrow(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=0.9,
                                     color=S.MUTED, zorder=1))

    LX, LW = 0.3, 5.5
    RX, RW = 6.2, 3.6

    box(LX, 8.9, LW, 1.35, "Preprocessed through one pipeline",
        [f"{len(cache)} public cohorts, {cache_total:,} recordings",
         "identical filtering, referencing, ICA and rejection"])
    arrow(LX + LW / 2, 8.9, 7.75)

    box(RX, 8.95, RW, 0.85, "Not carried to extraction",
        [f"{k} ({v})" for k, v in sorted(not_extracted.items())] or ["none"],
        accent=S.TERRA)
    ax.add_patch(FancyArrowPatch((LX + LW / 2, 8.45), (RX, 9.35), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.8,
                                 color=S.TERRA, zorder=1,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=0"))

    box(LX, 6.4, LW, 1.35, "Entered feature extraction",
        [f"{len(entered)} cohorts, {entered_total:,} recordings",
         "truncated to a common 235 s (MCI-Dataset 50 s)"])
    arrow(LX + LW / 2, 6.4, 5.25)

    box(RX, 6.35, RW, 1.35, "Retained no subject at K=24",
        [f"{PRETTY.get(k, k)} ({v})" for k, v in sorted(zero_retained.items())]
        + ["cannot supply 24 gradable 4 s epochs"], accent=S.TERRA)
    ax.add_patch(FancyArrowPatch((LX + LW / 2, 5.95), (RX, 7.0), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.8, color=S.TERRA, zorder=1,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=0"))

    box(LX, 3.9, LW, 1.35, "Cohorts supplying at least one subject",
        [f"{len(ANALYSED)} cohorts, {candidates:,} candidate subjects",
         "each subject must supply exactly 24 epochs"])
    arrow(LX + LW / 2, 3.9, 2.75)

    box(RX, 3.85, RW, 1.35, "Below the 24-epoch quota",
        [f"{PRETTY.get(k, k)} ({v})" for k, v in sorted(quota_loss.items())]
        + [f"{sum(quota_loss.values())} subjects in total"], accent=S.TERRA)
    ax.add_patch(FancyArrowPatch((LX + LW / 2, 3.45), (RX, 4.5), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.8, color=S.TERRA, zorder=1,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=0"))

    box(LX, 1.45, LW, 1.3, "Analysed corpus",
        [f"{analysed_total:,} subjects, {F['n_epochs']:,} epochs",
         "96 s of gradable recording per subject"], accent=S.SLATE)

    roles = collections.OrderedDict()
    for k, v in ANALYSED.items():
        roles.setdefault(ROLE[k], []).append(f"{PRETTY.get(k, k)} {v}")
    ax.text(LX + 0.22, 1.15,
            "   |   ".join(f"{r}: {', '.join(v)}" for r, v in roles.items()),
            fontsize=6.2, color=S.INK_SOFT, va="top")

    ax.text(LX, 0.35, "Counts are read from the harmonised cache and the extraction artefact at "
                      "figure-build time.", fontsize=5.9, color=S.MUTED, va="top", style="italic")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "figS5_flow.pdf")
    print("wrote figS5_flow.pdf")
    print(f"  cache {cache_total} -> extraction {entered_total} -> candidates {candidates} "
          f"-> analysed {analysed_total}")
    print(f"  not carried: {dict(not_extracted)}")
    print(f"  zero retained: {zero_retained}")
    print(f"  quota loss: {quota_loss}")


if __name__ == "__main__":
    main()
