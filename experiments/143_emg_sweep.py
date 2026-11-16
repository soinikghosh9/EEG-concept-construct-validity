"""Does the construct-validity verdict depend on the exact 20 Hz analysis ceiling?

The three concepts that fail construct validation do so because they integrate power through the
band in which scalp electromyogram dominates. The remedy is an analysis ceiling, fixed a priori at
20 Hz from the neuromuscular-blockade literature. That justification is sound, but a reader is
entitled to ask what would have happened at 15, 25 or 30 Hz -- and if the verdicts flipped at
exactly 20, the ceiling would be a tuned parameter wearing a citation.

This collects the validity battery across cutoffs already extracted by
`127_extract_harmonised.py --emg-hz`, and reports how each concept's verdict moves. What should be
seen, if the argument holds:

  * the aperiodic exponent and spectral median clean up progressively as the ceiling falls, with
    no threshold at which they suddenly change verdict;
  * spectral entropy stays artefact-dominated at every cutoff, because broadband noise flattens
    whatever band it is given -- that is the a priori argument, and a sweep is the only way to
    show it rather than assert it;
  * criterion validity against MMSE does not collapse as the ceiling falls, which would indicate
    the restriction was removing signal along with the artefact.

A verdict that holds across the grid is robust. A verdict that exists only at 20 Hz is not a
finding, and this script is designed to be able to say so.

    python experiments/143_emg_sweep.py --mode channel
Out: outputs/results/audit/emg_sweep_<mode>.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, safe_auc  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402
from src.data.feature_table import check_table  # noqa: E402

FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "audit"
RAW_ON = ROOT / "datasets" / "openneuro_ds004504_raw"
GRID = (15, 20, 25, 30)
WATCH = ("aperiodic_exponent", "spectral_edge", "spectral_entropy",
         "theta_alpha_ratio", "rel_theta", "pdr_frequency")
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    r = lambda v: np.argsort(np.argsort(v)).astype(float)          # noqa: E731
    return float(np.corrcoef(r(a[ok]), r(b[ok]))[0, 1])


def williams(r12, r13, r23, n):
    from scipy.stats import t as tdist
    if any(v is None for v in (r12, r13, r23)) or n < 10:
        return None, None
    det = 1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2.0 * r12 * r13 * r23
    num = (r12 - r13) * np.sqrt((n - 1) * (1.0 + r23))
    den = np.sqrt(2.0 * ((n - 1) / (n - 3)) * det + ((r12 + r13) ** 2 / 4.0) * (1.0 - r23) ** 3)
    if not np.isfinite(den) or den <= 0:
        return None, None
    t = float(num / den)
    return t, float(2.0 * (1.0 - tdist.cdf(abs(t), df=n - 3)))


def read_mmse():
    out = {}
    p = RAW_ON / "participants.tsv"
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            k = next((k for k in row if k.strip().upper() == "MMSE"), None)
            pid = (row.get("participant_id") or "").strip()
            if k and pid:
                try:
                    out[pid] = float(row[k])
                except (TypeError, ValueError):
                    pass
    return out


def fmt(v, nd=3):
    return "  n/a " if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="channel")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from src.models.v6 import CONCEPT_NAMES, aggregate_subject

    mmse_map = read_mmse()
    report = {"mode": args.mode, "grid_hz": list(GRID), "cutoffs": {}}
    table = {}

    for hz in GRID:
        tag = args.mode if hz == 20 else f"{args.mode}_emg{hz}"
        d = FEAT / tag
        if not (d / "window_concepts.npy").exists():
            print(f"[skip] {hz} Hz: {d} not extracted")
            continue
        # every array in this directory is joined by row index; a stale one would silently
        # pair features with the wrong windows
        check_table(d)
        C_win = np.load(d / "window_concepts.npy")
        meta = np.load(d / "meta.npz", allow_pickle=False)
        groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]
        subs = np.unique(groups)
        idxs = [np.flatnonzero(groups == s) for s in subs]
        sub_coh = np.array([cohort[i[0]] for i in idxs])
        # No three-class mapping here: every statistic below is computed on the MMSE subset, which
        # is the 88 OpenNeuro subjects in every arm of the sweep. Mapping labels for the whole
        # corpus served nothing and crashed once the queue harmonised cohorts carrying label
        # vocabularies ("Healthy") that this study's mapping does not define.
        C = np.stack([aggregate_subject(C_win[i]) for i in idxs])
        art = C[:, CONCEPT_NAMES.index("artefact_proxy")]
        m_on = np.array([mmse_map.get(str(s), np.nan) for s in subs], dtype=float)
        on = np.isfinite(m_on)

        row = {}
        for name in CONCEPT_NAMES:
            if name == "artefact_proxy":
                continue
            j = CONCEPT_NAMES.index(name)
            vo, ao, mo = C[on, j].copy(), art[on], m_on[on].copy()
            r_art = spearman(vo, ao)
            if r_art is not None and r_art < 0:
                vo = -vo
            r12, r13 = spearman(vo, ao), spearman(vo, mo)
            if r13 is not None and r13 < 0:
                mo = -mo
                r13 = spearman(vo, mo)
            t, p = williams(r12, r13, spearman(ao, mo), int(on.sum()))
            diff = None if (r12 is None or r13 is None) else r12 - r13
            verdict = ("unresolvable" if p is None or p >= 0.05
                       else "ARTEFACT" if diff and diff > 0 else "criterion")
            row[name] = {"rho_artefact": spearman(C[on, j], art[on]),
                         "rho_mmse": spearman(C[on, j], m_on[on]),
                         "williams_p": p, "verdict": verdict}
        report["cutoffs"][str(hz)] = {"n_with_mmse": int(on.sum()), "concepts": row}
        table[hz] = row
        print(f"{hz} Hz: n={int(on.sum())}, "
              f"{sum(1 for v in row.values() if v['verdict'] == 'ARTEFACT')} artefact-dominated")

    if not table:
        raise SystemExit("nothing extracted; run 127 with --emg-hz first")

    hzs = sorted(table)
    print(f"\n{'='*88}\nVerdict across the grid (n=88 with MMSE)")
    print(f"  {'concept':22s} " + "  ".join(f"{h:>13d} Hz" for h in hzs))
    for name in WATCH:
        if name not in table[hzs[0]]:
            continue
        cells = []
        for h in hzs:
            r = table[h][name]
            cells.append(f"{r['rho_artefact']:+.2f}/{r['rho_mmse']:+.2f} "
                         f"{'A' if r['verdict'] == 'ARTEFACT' else ('C' if r['verdict'] == 'criterion' else '-')}")
        print(f"  {name:22s} " + "  ".join(f"{c:>16s}" for c in cells))
    print("\n  cells are rho(artefact)/rho(MMSE), then A=artefact-dominated, "
          "C=criterion-dominated, -=unresolvable")

    stable = [n for n in table[hzs[0]]
              if len({table[h][n]["verdict"] for h in hzs}) == 1]
    flips = [n for n in table[hzs[0]] if n not in stable]
    report["verdict_stable_across_grid"] = stable
    report["verdict_changes_across_grid"] = flips
    print(f"\n  verdict identical at every cutoff: {len(stable)}/{len(table[hzs[0]])}")
    if flips:
        print(f"  changes verdict somewhere: {', '.join(flips)}")

    # Named for the scaling arm, not a corpus tag, because the sweep spans the four cutoff
    # tables of one arm rather than one alternative corpus.
    dest = write_result(OUT / f"emg_sweep_{args.mode}.json", report,
                        script="143_emg_sweep.py")
    print(f"\nwrote {dest}")
    print("\nIf a concept's verdict holds at every cutoff, the 20 Hz choice is not doing the work. "
          "If a\nverdict exists only at 20 Hz, that concept's result must be reported as "
          "cutoff-dependent.")


if __name__ == "__main__":
    main()
