"""What a clinician should be able to look at behind the score.

The report panel shows how much each concept moved the decision. That answers "why did the model
say this" but not "what does this recording look like", which is the question a neurophysiologist
actually asks before accepting or rejecting a machine's opinion. This figure supplies the second:
the scalp distribution of slowing, the regional band profile against a normative range, and the
concept deviations with the model's actually-used concepts marked.

Everything is drawn from the same 123-dimensional spectral block the comparator models use, which
is 19 channels by 5 bands plus regional means, so nothing here requires data the pipeline does not
already produce.

Normative range is the interquartile range of the training cohort's cognitively normal subjects. It
is a reference distribution, not a diagnostic threshold, and is labelled as such in the figure.

    python experiments/159_clinical_view.py
Out: jne_build/Figs/fig3_clinical.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import _figstyle_2026 as S  # noqa: E402

FIGS = ROOT / "jne_build" / "Figs"
FEAT = ROOT / "outputs" / "harmonised_features"
CH = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
      'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
REGION = ["frontal", "central", "temporal", "parietal", "occipital"]
DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}


#: Display names, so the panel a clinician reads matches the tables in the paper rather than the
#: identifiers in the code.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    from sklearn.model_selection import StratifiedKFold
    from src.models.v6 import SignConstrainedLogistic
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
    S.apply(plt)

    T = tb.subject_table(args.tag)
    names, X, coh, cls = T["names"], T["X_concept"], T["cohort"], T["cls"]
    Sp = T["X_spectral"]
    if Sp is None:
        print("[skip] no spectral block"); return
    tr = coh == "CAUEEG"
    y1 = (cls != "CN").astype(int)
    ctrl = tr & (cls == "CN")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X[tr], y1[tr]))
    fits = [SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(X[tr][a], y1[tr][a])
            for a, _ in folds]
    w = np.mean([np.asarray(getattr(f, "coef_", getattr(f, "w", None))).ravel() for f in fits], 0)
    used = np.abs(w) >= 1e-8

    def representative(mask):
        """The subject nearest the cohort's median score, so the panel shows a typical case."""
        idx = np.flatnonzero(mask)
        pr = np.mean([f.predict_proba(X[idx])[:, 1] for f in fits], axis=0)
        k = int(np.argmin(np.abs(pr - np.median(pr))))
        return int(idx[k]), float(pr[k])

    # Two cases. The first is Alzheimer's at the blind site, the contrast the tool is strongest on.
    # The second is from a cohort that contains no controls at all, which is the situation a
    # deployed tool meets on a ward: there the panel is the only reference the reader has.
    pick, prob = representative((coh == "ds004504_raw") & (cls == "AD"))
    CASES = [(pick, prob, "Alzheimer's, blind site")]
    sc_mask = (coh == "P-ADIC")
    if sc_mask.sum() >= 5:
        pick2, prob2 = representative(sc_mask)
        CASES.append((pick2, prob2, "mild cognitive impairment,\nsingle-class cohort"))

    chan = Sp[:, :95].reshape(-1, 19, 5)          # [subject, channel, band]
    reg = Sp[:, 95:120].reshape(-1, 5, 5)         # [subject, region, band]

    info = mne.create_info([c.replace("T3", "T7").replace("T4", "T8")
                            .replace("T5", "P7").replace("T6", "P8") for c in CH],
                           sfreq=256., ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # Topographies on the top row, the concept profile across the full width below. The previous
    # layout put the profile in a narrow right-hand column, where its labels were drawn over the
    # bars.
    fig = plt.figure(figsize=(7.3, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.7], hspace=0.16, wspace=0.24)

    # ---- topographies: patient against the control median
    for k, band in enumerate(("theta", "alpha")):
        bi = BANDS.index(band)
        pat = chan[pick, :, bi]
        nor = np.median(chan[ctrl][:, :, bi], axis=0)
        # A 0-to-max scale drove both panels to one flat colour, because the two differ by a
        # factor of five in this band. The scale spans the values actually present in the pair,
        # so the same colour means the same relative power in both heads.
        lo = float(min(pat.min(), nor.min()))
        hi = float(max(pat.max(), nor.max()))
        for j, (v, ttl) in enumerate(((nor, "normal median"), (pat, "Alzheimer's case"))):
            ax = fig.add_subplot(gs[0, 2 * k + j])
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(lo, hi), contours=2, sensors=True)
            ax.set_title("%s\nrelative %s" % (ttl, band), fontsize=6.6, pad=4)
            if j == 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
                cb.ax.tick_params(labelsize=5.4)
                cb.outline.set_visible(False)

    # ---- concept deviation profile, one column per case, on a shared scale
    med = np.median(X[ctrl], axis=0)
    iqr = np.subtract(*np.percentile(X[ctrl], [75, 25], axis=0))
    iqr[iqr == 0] = np.nan
    Zs = [(X[c[0]] - med) / iqr for c in CASES]
    order = np.argsort(Zs[0])                     # one ordering, so rows align across cases
    lim = float(np.nanmax(np.abs(np.concatenate(Zs)))) * 1.12
    half = 2 if len(CASES) > 1 else 4
    for ci, ((_, pr, ttl), z) in enumerate(zip(CASES, Zs)):
        ax = fig.add_subplot(gs[1, ci * half:(ci + 1) * half])
        yy = np.arange(len(order))
        cols = [(S.TERRA if z[j] > 0 else S.SLATE) if used[j] else S.STONE for j in order]
        ax.barh(yy, z[order], height=0.62, color=cols)
        ax.axvspan(-1, 1, color=S.HAIR, alpha=0.45, zorder=0)
        ax.axvline(0, color=S.MUTED, lw=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(yy)
        if ci == 0:
            ax.set_yticklabels([CONCEPT_NICE.get(names[j], names[j].replace("_", " "))
                                for j in order], fontsize=6.2)
            for k, j in enumerate(order):
                if not used[j]:
                    ax.annotate("unused", (0.0, k), xytext=(-56, 0),
                                textcoords="offset points", va="center", ha="left",
                                fontsize=5.2, color=S.MUTED, annotation_clip=False)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xlabel("deviation from normal, robust z", fontsize=6.6)
        ax.set_title("%s\np(impaired) %.2f" % (ttl, pr), loc="left", fontsize=7.0)
        S.hgrid(ax, "x")

    fig.suptitle("What the recording looks like behind the score", x=0.02, ha="left",
                 fontsize=8.4)
    fig.subplots_adjust(left=0.20)
    fig.text(0.02, 0.005,
             "Grey bars are concepts the model assigns zero weight; they describe the recording "
             "but do not enter the decision.", fontsize=5.8, color=S.MUTED)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig3_clinical.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "_fig8_preview.png", dpi=150, bbox_inches="tight")
    print("wrote fig3_clinical.pdf  (subject index %d, p=%.3f, %d/%d concepts used)"
          % (pick, prob, int(used.sum()), len(names)))


if __name__ == "__main__":
    main()
