"""Figure: the blinded reader study (main-text figure of the construct paper).

Reads only the artefacts written by 266_reader_study_analysis.py, so the figure cannot drift from
the numbers quoted in the text.

    python experiments/267_fig_reader_study.py
Out: jne_build_revised/Figs/fig10_readers.pdf (+ .png)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sciplots as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results" / "validity"
OUT = ROOT / "jne_build_revised" / "Figs" / "fig10_readers"


def main():
    r = json.loads((RES / "reader_study_results.json").read_text())
    t = pd.read_csv(RES / "reader_study_per_recording.csv")
    ir, re_ = r["inter_reader"], r["reader_vs_estimator"]

    sp.setup("nature")
    R1, R2 = sp.HOUSE[0], sp.HOUSE[1]
    fig = sp.figure(140, 125)
    (a, b), (c,) = sp.grid(fig, [[1, 1.25], [1]], height_ratios=[1.15, 1])

    # ---------------------------------------------------------------- a. PDR frequency
    lo, hi = 4.0, 12.0
    a.plot([lo, hi], [lo, hi], color=sp.AXIS, lw=0.5, ls=(0, (2, 2)), zorder=0)
    rng = np.random.default_rng(0)
    for col, colr, nm in [("R1", R1, "reader 1"), ("R2", R2, "reader 2")]:
        num = t[f"{col}_pdr"].notna()
        a.scatter(t.loc[num, "pdr_frequency"], t.loc[num, f"{col}_pdr"] + rng.uniform(-.08, .08, num.sum()),
                  s=6, color=colr, alpha=0.75, lw=0, zorder=2)
        ab = t[f"{col}_pdr_absent"] == 1
        a.scatter(t.loc[ab, "pdr_frequency"], np.full(ab.sum(), 3.45) + (0.0 if col == "R1" else -0.25),
                  s=9, marker="x", color=colr, lw=0.7, zorder=3)
    a.axhline(3.8, color=sp.AXIS, lw=0.4)
    a.set_xlim(lo, hi); a.set_ylim(3.0, hi)
    a.set_yticks([3.35, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 ["none", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    a.set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12])
    a.set_xlabel("estimator: PDR frequency (Hz)")
    a.set_ylabel("reader: PDR frequency (Hz)")
    a.set_aspect(1.0, adjustable="box")
    sp.key(a, [(sp.h_marker(R1, "o", ms=2.4), "reader 1"), (sp.h_marker(R2, "o", ms=2.4), "reader 2"),
               (sp.h_marker(sp.INK, "x", ms=3), "reader: no rhythm"),
               (sp.h_line(sp.AXIS, lw=0.5, ls=(0, (2, 2))), "identity")], loc="top")

    # ---------------------------------------------------------------- b. inter-reader agreement
    rows = [("PDR frequency (ICC)", ir["pdr_frequency"]["icc_a1"], *ir["pdr_frequency"]["icc_ci"]),
            ("PDR well formed (κw)", ir["wf"]["kappa"], *ir["wf"]["kappa_ci"]),
            ("slowing present (κ)", ir["slowing_present"]["kappa"], *ir["slowing_present"]["kappa_ci"]),
            ("slowing grade (κw)", ir["slow"]["kappa"], *ir["slow"]["kappa_ci"]),
            ("global impression (κw)", ir["glob"]["kappa"], *ir["glob"]["kappa_ci"]),
            ("posterior maximal (κ)", ir["pmax"]["kappa"], *ir["pmax"]["kappa_ci"]),
            ("no rhythm identified (κ)", ir["pdr_absent"]["kappa"], *ir["pdr_absent"]["kappa_ci"]),
            ("muscle artefact (κw)", ir["art"]["kappa"], *ir["art"]["kappa_ci"]),
            ("drowsiness (κw)", ir["drow"]["kappa"], *ir["drow"]["kappa_ci"])]
    sp.forest(b, [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows],
              [x[3] for x in rows], ref=0.0, ref_label="chance", color=sp.HOUSE[2],
              highlight=[x[0] for x in rows], value_label="agreement [95% CI]", d=2, key_loc=None)
    b.set_xlim(-0.1, 1.0)
    b.set_xlabel("reader 1 vs reader 2 agreement")

    # ---------------------------------------------------------------- c. reader vs estimator
    pairs = [("slowing grade · slowing ratio", "slow~slowing_ratio|consensus"),
             ("slowing grade · theta–alpha ratio", "slow~theta_alpha_ratio|consensus"),
             ("slowing grade · relative theta", "slow~rel_theta|consensus"),
             ("posterior maximal · amplitude form", "pmax~amp_posterior_maximality|consensus"),
             ("posterior maximal · relative form (reported)", "pmax~posterior_maximality|consensus"),
             ("PDR well formed · PDR prominence", "wf~pdr_prominence|consensus"),
             ("muscle artefact · artefact instrument", "art~artefact_proxy|consensus"),
             ("drowsiness · theta–alpha ratio", "drow~theta_alpha_ratio|consensus")]
    est = [re_[k]["spearman"] for _, k in pairs]
    lo_ = [re_[k]["ci"][0] for _, k in pairs]
    hi_ = [re_[k]["ci"][1] for _, k in pairs]
    hl = [nm for (nm, _), l in zip(pairs, lo_) if l > 0]
    sp.forest(c, [p[0] for p in pairs], est, lo_, hi_, ref=0.0, ref_label="no association",
              highlight=hl, color=sp.HOUSE[0], value_label="Spearman's ρ [95% CI]", d=2, key_loc=None)
    c.set_xlim(-0.2, 1.0)
    c.set_xlabel("association of the readers' mean grade with the estimator (Spearman's ρ)")
    sp.key(c, [(sp.h_marker(sp.HOUSE[0], "s", ms=3), "95% CI excludes zero"),
               (sp.h_marker(sp.GREY, "o", ms=2.6), "95% CI includes zero"),
               (sp.h_line(sp.AXIS, lw=0.5, ls=(0, (2, 2))), "no association")], loc="top")

    problems = sp.finish(fig, [a, b, c], str(OUT))
    print("problems:", problems)


if __name__ == "__main__":
    main()
