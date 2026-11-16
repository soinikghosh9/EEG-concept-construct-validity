"""Acquisition floors for every candidate cohort, with provenance.

Table 5 of the manuscript reports, per candidate evaluation site, the AUC obtainable from recording
duration alone and from surviving epoch count alone -- what a model could score without reading the
brain. Those numbers were not traceable to any current artefact, and the cohort sizes printed
beside them did not reconcile: the table gave the memory-clinic site 159 where the pipeline says
166, and the olfactory-task set 28 where the cohort it names supplies 35.

This recomputes both floors from the pipeline's own record and prints the census beside them, so
the table can be sourced and the counts reconciled. Duration comes from the preprocessing cache
(`outputs/harmonised/*.json`, one file per recording, carrying `qc.duration_s`). Epoch count comes
from the pre-quota extraction arm, in which subjects still contribute unequal numbers of windows;
the analysed arm equalises them at 24 by construction and would report 0.5 for every cohort.

A floor is defined only where a cohort contains both classes and the quantity varies within it: a
single-class cohort admits no AUC, and a cohort whose recordings are all the same length admits no
duration AUC. Those cases are reported as null rather than as 0.5.

    python experiments/168_acquisition_floors.py
Out: outputs/results/audit/acquisition_floors.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

CACHE = ROOT / "outputs" / "harmonised"
FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"

#: the arm in which subjects still carry unequal window counts
PREQUOTA = "channel_all"

#: cohort label -> how the manuscript names it
PRETTY = {"ds004504_raw": "Dementia site (OpenNeuro)", "GENEEG": "Prodromal site (GENEEG)",
          "MCI_Dataset": "Memory-clinic site", "AlzEEG": "Short-recording site",
          "Mendeley": "Olfactory-task set", "MISP": "MISP", "P-ADIC": "P-ADIC",
          "CAUEEG": "Training cohort (CAUEEG)"}

#: Control labels, by cohort vocabulary. An unrecognised label must abort rather than default to
#: "impaired": treating AlzEEG's 23 `Healthy` subjects as cases silently turned a two-class cohort
#: into a single-class one and suppressed both of its floors.
CONTROL = {"CN", "C", "CONTROL", "Control", "Normal", "normal", "HC", "Healthy", "healthy"}
IMPAIRED = {"AD", "A", "FTD", "F", "MCI", "DEM", "Dementia", "dementia"}


def auc(y, x):
    """Mann-Whitney AUC of a single continuous quantity; None when it cannot be defined."""
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return None
    from scipy.stats import rankdata
    r = rankdata(x)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=PREQUOTA)
    args = ap.parse_args()
    from src.utils.provenance import write_result

    # ---- duration, from the preprocessing cache -------------------------------------------
    dur, grp, coh = {}, {}, {}
    for f in glob.glob(str(CACHE / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:                                              # noqa: BLE001
            continue
        s = d.get("subject")
        if not s:
            continue
        dur[s] = (d.get("qc") or {}).get("duration_s")
        grp[s] = d.get("group")
        coh[s] = d.get("cohort")
    print("cache: %d recordings across %d cohorts" % (len(dur), len(set(coh.values()))))

    # ---- epoch count, from the pre-quota extraction arm ------------------------------------
    meta = np.load(FEAT / args.arm / "meta.npz", allow_pickle=False)
    nwin = collections.Counter(meta["groups"].tolist())
    print("pre-quota arm '%s': %d subjects" % (args.arm, len(nwin)))

    unknown = sorted({g for g in grp.values() if g not in CONTROL and g not in IMPAIRED})
    if unknown:
        raise SystemExit("unmapped group labels, refusing to guess: %s" % unknown)

    cohorts = sorted(set(coh.values()))
    report = {"arm": args.arm, "n_cache": len(dur), "control_labels": sorted(CONTROL),
              "cohorts": {}}

    print("\n%-28s %6s %6s %8s %10s %10s" % ("cohort", "n", "n_pos", "n_neg",
                                             "duration", "epochs"))
    total = 0
    for c in cohorts:
        subs = [s for s in dur if coh[s] == c]
        total += len(subs)
        lab = [grp[s] for s in subs]
        y = np.array([0 if (l in CONTROL) else 1 for l in lab])
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())

        a_dur = auc(y, [dur[s] for s in subs]) if nneg and npos else None
        have = [s for s in subs if s in nwin]
        a_ep = None
        if have and nneg and npos:
            yy = np.array([0 if (grp[s] in CONTROL) else 1 for s in have])
            a_ep = auc(yy, [nwin[s] for s in have])

        durs = [dur[s] for s in subs if dur[s] is not None]
        report["cohorts"][c] = {
            "name": PRETTY.get(c, c), "n": len(subs), "n_pos": npos, "n_neg": nneg,
            "classes": sorted(set(lab)),
            "duration_auc": a_dur, "epoch_count_auc": a_ep,
            "n_reaching_extraction": len(have),
            "duration_median_s": float(np.median(durs)) if durs else None,
            "duration_constant": bool(durs and np.ptp(durs) < 1e-6)}
        print("%-28s %6d %6d %8d %10s %10s"
              % (PRETTY.get(c, c), len(subs), npos, nneg,
                 "---" if a_dur is None else "%.3f" % a_dur,
                 "---" if a_ep is None else "%.3f" % a_ep))

    report["n_total_checked"] = total
    print("\ntotal accounted: %d (cache holds %d)" % (total, len(dur)))
    if total != len(dur):
        print("  [warn] census does not reconcile against the cache")

    dest = write_result(OUT / "acquisition_floors.json", report, script=Path(__file__).name)
    print("wrote %s" % dest)


if __name__ == "__main__":
    main()
