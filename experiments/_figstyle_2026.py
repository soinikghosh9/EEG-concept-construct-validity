"""Shared visual language for the manuscript figures.

One place for the palette, typography and axis treatment, so every panel reads as part of the same
document rather than as a collection of plots made on different days. The choices below are
deliberate rather than decorative:

* **A warm off-white ground, not pure white.** Pure white against black text is the highest
  contrast a page can carry and is tiring across a figure-dense paper; a very slightly warm ground
  softens it without looking tinted in print.
* **Muted, desaturated hues.** Saturated primaries fight each other and make the eye rank
  categories by vividness rather than by value. Every colour here sits at similar perceived
  lightness so that no series shouts.
* **Two semantic colours and everything else neutral.** Terracotta marks a problem, slate marks a
  result, and grey carries context. A reader should be able to find the finding without a legend.
* **No top or right spine, hairline axes, ticks outside.** The data should be the darkest thing on
  the panel.
* **Direct labels wherever they fit.** A legend forces a saccade between key and mark; a label on
  the mark does not.

Colours are checked to remain distinguishable in greyscale and under the common forms of colour
vision deficiency, since terracotta and slate differ in lightness as well as hue.
"""
from __future__ import annotations

# ---- ground and ink ---------------------------------------------------------------------
PAPER = "#FFFFFF"        # white page. Journals print on white and reviewers read on white;
                         # a tinted ground reads as a rendering artefact rather than a choice.
PANEL = "#FFFFFF"        # panel interior where a lift is wanted
INK = "#26262A"          # primary text
INK_SOFT = "#55555C"     # secondary text
MUTED = "#9B9BA2"        # axis furniture, tick labels
HAIR = "#DEDDD7"         # gridlines, hairlines

# ---- semantic ---------------------------------------------------------------------------
SLATE = "#4A6FA5"        # a result, a model, the "our" series
TERRA = "#C0663C"        # a problem: artefact, failure, exclusion
SAGE = "#7C9A6B"         # a success, a repaired quantity
SAND = "#D4A659"         # a third category
PLUM = "#8B6B8F"         # a fourth category
STONE = "#B4B3AC"        # neutral filler

SEQ = [SLATE, TERRA, SAGE, SAND, PLUM, STONE]

#: Component classes, kept consistent across every panel that shows them.
COMPONENT = {"brain": SLATE, "eye": TERRA, "muscle": SAND,
             "line": "#8E8E94", "chan": "#C3C2BB", "other": "#DEDDD7"}


def apply(plt):
    plt.rcParams.update({
        "figure.dpi": 240,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,

        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Segoe UI", "Helvetica Neue", "Arial",
                            "DejaVu Sans"],
        "font.size": 7.6,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.4,
        "axes.titleweight": "regular",
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,

        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8,
        "axes.labelpad": 5,

        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "grid.color": HAIR,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,

        "legend.frameon": False,
        "legend.handlelength": 1.1,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,

        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
    })


def strip(ax, x=True, y=False, left=True, bottom=True):
    """Remove furniture that is not carrying information."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if x:
        ax.xaxis.grid(False)
    if y:
        ax.yaxis.grid(False)
    return ax


def hgrid(ax, axis="x"):
    """A faint reference grid, drawn behind everything."""
    ax.set_axisbelow(True)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, color=HAIR, lw=0.7)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, color=HAIR, lw=0.7)
    return ax


def label_end(ax, x, y, text, color=INK, dx=0.006, size=6.8, weight="regular", va="center"):
    """Direct label at the end of a mark, in place of a legend entry."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset fontsize",
                color=color, fontsize=size, va=va, ha="left", fontweight=weight)


def panel_tag(fig, ax, letter):
    """Panel letters in the top-left, outside the axes, unobtrusive."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-26, 10), textcoords="offset points",
                fontsize=9.0, fontweight="semibold", color=INK, va="bottom", ha="left")


def caption(ax, text, y=-0.30, size=6.6):
    ax.annotate(text, xy=(0, y), xycoords="axes fraction", fontsize=size,
                color=INK_SOFT, va="top", ha="left", wrap=True)


def sequential(plt, light="#F4F3EF", dark=SLATE):
    """A sequential map built from the document's own palette.

    Matplotlib's stock sequential maps (BuPu and the like) introduce a saturated violet that
    belongs to no other mark in the paper, so a heatmap drawn with one reads as imported from a
    different document. This ramps the page ground to the slate already used for every result.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper_seq", [light, dark])


def diverging(plt, low=TERRA, mid="#F4F3EF", high=SLATE):
    """Two-sided version, for quantities with a meaningful zero."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper_div", [low, mid, high])
