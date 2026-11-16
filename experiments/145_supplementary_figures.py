"""Supplementary figures.

The main text carries the six panels that make the argument. These carry the evidence a reader
would want in order to check it, and the material that is informative without being load-bearing.

    S1  the validation battery as a profile per concept  (radar)
    S2  where the interpretability cost actually sits    (waterfall)
    S3  capacity against performance across ten architectures
    S4  spectral profile of each cohort, which is why one of them cannot be transferred to
    S5  transport: what is achievable at the prodromal site and what is reached

A note on S1. Radar plots are easy to misuse: with many overlaid series they become unreadable, and
because area grows with the square of the radius they exaggerate differences. They are used here
for the one thing they do well -- comparing a small number of items across several commensurable
axes at once -- with three concepts, one from each outcome of the battery, and axes all scaled to
a common zero-to-one "better is outward" convention. The underlying numbers are in table S4; the
figure is a summary of shape, not a substitute for them.

    python experiments/145_supplementary_figures.py
Out: jne_build/Figs/figS*.pdf
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

# Must match the arm the manuscript tables report; see 144_manuscript_figures.py. Drawing S3 from
# the `channel` arm while table 2 reports `channel_clean` is what made the supplement's capacity
# figure disagree with the main table on the same two models.
TAG = "channel_clean"


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




# ------------------------------------------------------------------ S2 cost waterfall
def s2_cost(plt):
    # the harmonised decomposition (169), computed on the analysed 1,296 subjects; the older
    # cost_decomposition.json runs through the pre-harmonisation corpus and is superseded
    d = load(AUD, "cost_decomposition_harmonised.json")
    if not d:
        print("  [skip] S2"); return
    dec = d["decomposition"]
    steps = [("training unit", dec["training_unit_A0_minus_A1"]),
             ("bottleneck", dec["bottleneck_A1_minus_A2"]),
             ("linearity", dec["linearity_A2_minus_A4"]),
             ("hierarchy", dec["hierarchy_A4_minus_A5"]),
             ("sign constraints", dec["sign_constraints_A5_minus_A6"])]
    start = d["arm_auc"]["A0_window_RF_spectral"]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(steps) + 2)
    running = start
    ax.bar(0, start, bottom=0, width=0.6, color=S.STONE)
    ax.annotate(f"{start:.3f}", (0, start), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=6.6, color=S.INK_SOFT)
    for i, (lbl, v) in enumerate(steps, start=1):
        bottom = running - v if v > 0 else running
        ax.bar(i, abs(v), bottom=bottom, width=0.6,
               color=S.TERRA if v > 0 else S.SAGE)
        ax.annotate(f"{-v:+.3f}", (i, bottom + abs(v)), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6.4,
                    color=S.TERRA if v > 0 else S.SAGE)
        running -= v
    ax.bar(len(steps) + 1, running, bottom=0, width=0.6, color=S.SLATE)
    ax.annotate(f"{running:.3f}", (len(steps) + 1, running), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=6.6, color=S.INK_SOFT)
    ax.set_xticks(x)
    ax.set_xticklabels(["forest\n(spectral)"] + [s[0] for s in steps] +
                       ["concept model\n(constrained)"], fontsize=6.0, rotation=20,
                       ha="right", rotation_mode="anchor")
    ax.set_ylim(0.60, 0.92)
    ax.set_ylabel("AUC")
    S.hgrid(ax, "y")
    ax.set_title("the cost is the sign constraints, not the named concepts",
                 loc="left", x=-0.02)
    fig.subplots_adjust(bottom=0.30)
    save(fig, "figS1_cost.pdf"); plt.close(fig)




# ------------------------------------------------------------------ S4 spectral profile
def s4_spectra(plt):
    from src.data.feature_table import check_table
    feat = ROOT / "outputs" / "harmonised_features" / TAG
    if not (feat / "spectral.npy").exists():
        print("  [skip] S4"); return
    check_table(feat)   # spectral rows are indexed against the meta below
    Sp = np.load(feat / "spectral.npy", mmap_mode="r")
    meta = np.load(feat / "meta.npz", allow_pickle=False)
    g, co = meta["groups"], meta["cohort"]
    subs = np.unique(g)
    idxs = [np.flatnonzero(g == s) for s in subs]
    coh = np.array([co[i[0]] for i in idxs])
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    show = [("CAUEEG", "CAUEEG (training)", S.SLATE),
            ("ds004504_raw", "OpenNeuro (test)", S.SAND),
            ("GENEEG", "GENEEG (test)", S.TERRA)]
    x = np.arange(len(bands))
    for c, lbl, col in show:
        m = coh == c
        if m.sum() < 10:
            continue
        rows = []
        for i in np.flatnonzero(m):
            v = np.asarray(Sp[idxs[i]][:, :95], dtype=np.float64).reshape(len(idxs[i]), 19, 5)
            rows.append(np.median(v, axis=0).mean(0))
        R = np.stack(rows)
        med = np.median(R, axis=0)
        ax.plot(x, med, "-o", color=col, ms=4.5, lw=1.5)
        S.label_end(ax, x[-1], med[-1], lbl, color=col, dx=0.25)
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=7.0)
    ax.set_xlim(-0.2, 6.2)
    ax.set_yscale("log")
    ax.set_ylabel("relative band power (median)")
    S.hgrid(ax, "y")
    ax.set_title("one cohort's spectrum barely overlaps the others", loc="left", x=-0.02)
    save(fig, "figS3_spectra.pdf"); plt.close(fig)


# ------------------------------------------------------------------ S5 transport
def s5_transport(plt):
    b = load(AUD, "binary_heads.json")
    if not b:
        print("  [skip] S5"); return
    # Ten bars per panel. At 3.0 in the rows are ~21 pt apart, so the three-line labels the
    # arm keys used to produce ("123 spectral" / "random forest" / "raw", one line each)
    # overlapped their neighbours. Two lines and a taller panel clear it with room to spare.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=False)

    def arm_label(key):
        feat, model, scale = [s.strip() for s in key.split("|")]
        scale = "site-std." if "standardised" in scale else scale
        return f"{feat}\n{model}, {scale}"

    for ax, (tname, label) in zip(axes, (("CN_vs_MCI", "prodromal site"),
                                         ("CN_vs_dementia", "dementia site"))):
        tg = b["targets"].get(tname)
        if not tg:
            continue
        arms = tg["arms"]
        keep = [(k, v) for k, v in arms.items() if "raw" in k or "standardised" in k]
        keep.sort(key=lambda kv: kv[1]["auc_mean"])
        y = np.arange(len(keep))
        for yi, (k, v) in zip(y, keep):
            col = S.SLATE if "concepts" in k else S.STONE
            ax.barh(yi, v["auc_mean"], height=0.55, color=col, zorder=3)
            # A bar ending near the ceiling line puts its value label on top of that line.
            # The halo keeps the number readable wherever it lands, without moving it away
            # from the bar it belongs to.
            ax.annotate(f"{v['auc_mean']:.3f}", (v["auc_mean"], yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.0,
                        color=S.INK_SOFT, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.12", fc=S.PAPER, ec="none"))
        ceil = max(v["ceiling"] for _, v in keep)
        ax.axvline(ceil, color=S.TERRA, lw=1.1, ls=(0, (4, 3)))
        ax.annotate(f"within-site ceiling {ceil:.3f}", (ceil, len(keep) - 0.3),
                    xytext=(-4, 0), textcoords="offset points", ha="right",
                    fontsize=6.2, color=S.TERRA)
        ax.set_yticks(y)
        ax.set_yticklabels([arm_label(k) for k, _ in keep], fontsize=5.8, linespacing=1.25)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.75, len(keep) - 0.25)
        ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC")
        S.hgrid(ax, "x")
        ax.set_title(label, loc="left", x=-0.02)
    fig.subplots_adjust(left=0.16, right=0.97, wspace=0.62, bottom=0.12, top=0.90)
    save(fig, "figS4_transport.pdf"); plt.close(fig)


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG,
                    help="analysis arm to draw from; must match the manuscript tables")
    TAG = ap.parse_args().tag
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)
    print(f"writing supplementary figures to {FIGS}  [arm: {TAG}]")
    # s1_battery is retired, not deleted: the radar it draws was cut from the supplement because
    # it restated table S4 in a form whose area exaggerates differences, and the main text's
    # failure-modes figure already shows the same three archetypes on real data. Running it would
    # write a figure nothing references. The function is kept so the decision stays readable.
    for fn in (s2_cost, s4_spectra, s5_transport):
        try:
            fn(plt)
        except Exception as e:                                         # noqa: BLE001
            import traceback
            print(f"  [fail] {fn.__name__}: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc(limit=1)
    print("done")


if __name__ == "__main__":
    main()
