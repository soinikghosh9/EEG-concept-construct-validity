"""Render blinded clinical EEG traces for the reader validation study (Phase 1).

The reading study is the one check in the construct-validation battery that no computation can
supply: whether a quantity the model calls "posterior dominant rhythm" is what a clinician reading
the same trace would call by that name. This script builds the material the readers grade.

Three properties matter and each is load-bearing:

1. **Traces are rendered from the ORIGINAL recordings, never from the analysis pipeline.** Every
   cohort loader per-channel robust-scales the signal, which destroys the microvolt scale and the
   inter-channel amplitude relationships. Amplitude is precisely what a reader grades, so rendering
   from preprocessed data would show them a trace with the evidence removed.

2. **Display is clinical, not scientific.** Longitudinal bipolar montage ("double banana") plus one
   referential page, 30 mm/s, 7 uV/mm, 0.5-70 Hz with mains notch. The geometry is enforced in
   millimetres on the page, not left to matplotlib's default scaling, so a reader can measure a
   frequency off the paper the way they would off a clinical viewer.

3. **Nothing on the page identifies the subject or the diagnosis.** Blind ID, age, sex, page number,
   montage and filter settings only. Diagnosis, cohort, MMSE, file name and model output all live
   in the sealed key, which is not opened until both readers have returned their sheets.

Usage
-----
  python experiments/146_export_reader_traces.py --manifest          # build + seal the sample
  python experiments/146_export_reader_traces.py --render 3          # first 3 as samples
  python experiments/146_export_reader_traces.py --render all        # the full study
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.harmonise import detect_mains          # noqa: E402
from src.utils.provenance import write_result        # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets"
OUT = ROOT / "READER_PACKET"
TRACES = OUT / "traces"
KEY = ROOT / "outputs" / "results" / "validity" / "reader_manifest_SEALED.csv"

# ---------------------------------------------------------------------------- display constants
MM_PER_S = 30.0          # clinical paper speed
UV_PER_MM = 7.0          # clinical sensitivity
PAGE_S = 10.0            # seconds per page
N_PAGES = 6              # bipolar pages -> 60 s of eyes-closed EEG
MIN_PAGES = 6            # below this a recording is excluded rather than padded
HP, LP = 0.5, 70.0       # display filters only

BIPOLAR = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]
CHAIN_BREAKS = {4, 8, 12, 16}          # blank gaps between montage chains
REFERENTIAL = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
               "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
POSTERIOR = ["O1", "O2", "P3", "P4"]

# CAUEEG event labels that disqualify a resting segment
DIRTY = ("photic", "hv", "hypervent", "artifact", "artefact", "chew", "swallow", "cough",
         "couch", "talk", "move", "sleep", "blink", "paused")


# ============================================================================= channel handling
def canon(name: str) -> str:
    """Normalise a channel label to old 10-20 naming (T3/T4/T5/T6, Fz/Cz/Pz)."""
    n = name.strip().upper().replace("-AVG", "").replace("-REF", "").replace("EEG ", "")
    alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    n = alias.get(n, n)
    if n in ("FZ", "CZ", "PZ"):
        return n.capitalize()
    if n.startswith("FP"):
        return "Fp" + n[2:]
    return n[0] + n[1:].lower() if len(n) > 2 and n[1].isalpha() else n


def as_frame(raw) -> tuple[dict[str, np.ndarray], float]:
    """Return {canonical channel: microvolt trace} and the sampling rate."""
    drop = [c for c in raw.ch_names if canon(c) in ("EKG", "ECG", "PHOTIC", "A1", "A2")]
    r = raw.copy().drop_channels(drop) if drop else raw.copy()
    data = r.get_data() * 1e6
    return {canon(c): data[i] for i, c in enumerate(r.ch_names)}, float(r.info["sfreq"])


# ============================================================================= segment selection
def caueeg_clean_intervals(serial: str, sfreq: float, n_samp: int) -> list[tuple[float, float]]:
    """Every eyes-closed interval free of photic, hyperventilation and noted artefact.

    CAUEEG alternates eyes open and closed roughly every thirty seconds, so most recordings hold
    no single minute of continuous eyes-closed EEG but several hundred seconds in total. Pages are
    therefore drawn from several intervals; each page is contiguous within one of them and carries
    its own timestamp, so the reader can see where the discontinuities fall.
    """
    f = DS / "caueeg-dataset" / "event" / f"{serial}.json"
    if not f.exists():
        return []
    ev = sorted([(int(s), str(lab)) for s, lab in json.loads(f.read_text())], key=lambda z: z[0])
    out = []
    for i, (s, lab) in enumerate(ev):
        if lab.strip().lower() != "eyes closed":
            continue
        end = min(ev[i + 1][0] if i + 1 < len(ev) else n_samp, n_samp)
        if any(s <= t < end and any(d in l.lower() for d in DIRTY) for t, l in ev):
            continue
        a, b = s / sfreq + 3.0, end / sfreq - 1.0      # guard the eye-closure transient
        if b - a >= PAGE_S:
            out.append((a, b))
    return out


def candidate_pages(cohort: str, subject: str, sfreq: float, n_samp: int) -> tuple[list[float], str]:
    """Every 10 s page of gradable eyes-closed EEG in the recording.

    There is deliberately no fallback for CAUEEG. A recording whose annotation yields too little
    clean eyes-closed EEG is excluded from the study and replaced, because the alternative --
    taking a fixed offset -- can land on eyes-open epochs or on photic stimulation, and a reader
    grading a photic driving response as an abnormal background would corrupt the endpoint the
    study exists to measure.
    """
    dur = n_samp / sfreq
    if cohort == "ds004504_raw":
        intervals = [(30.0, dur - 1.0)] if dur > 40 else [(0.0, dur - 1.0)]
        note = "eyes-closed task, continuous"
    else:
        intervals = caueeg_clean_intervals(subject, sfreq, n_samp)
        note = f"{len(intervals)} clean eyes-closed intervals"

    candidates = []
    for a, b in intervals:
        candidates.extend(a + k * PAGE_S for k in range(int((b - a) // PAGE_S)))
    return sorted(candidates), note


def pick_pages(cohort: str, subject: str, raw) -> tuple[list[float], str]:
    """Return the start time of each page, spread across the record, plus a provenance note."""
    sf = float(raw.info["sfreq"])
    candidates, note = candidate_pages(cohort, subject, sf, raw.n_times)
    if len(candidates) < MIN_PAGES:
        raise ValueError(f"only {len(candidates)} clean eyes-closed pages ({note}); "
                         f"{MIN_PAGES} required -- exclude and replace this recording")

    # spread the pages over the whole recording rather than taking the first minute: a reader
    # grading drowsiness needs to see late epochs as well as early ones
    if len(candidates) > N_PAGES:
        idx = np.linspace(0, len(candidates) - 1, N_PAGES).round().astype(int)
        pages = [candidates[i] for i in sorted(set(idx))]
    else:
        pages = candidates
    return pages, f"{note}; {len(candidates)} candidate pages"


def n_clean_pages(cohort: str, subject: str) -> int:
    """Cheap gradability screen used when the sample is drawn (header read only)."""
    try:
        if cohort == "ds004504_raw":
            f = (DS / "openneuro_ds004504_raw" / subject / "eeg" /
                 f"{subject}_task-eyesclosed_eeg.set")
            raw = mne.io.read_raw_eeglab(str(f), preload=False)
        else:
            raw = mne.io.read_raw_edf(str(DS / "caueeg-dataset" / "signal" / "edf" /
                                          f"{subject}.edf"), preload=False)
        return len(candidate_pages(cohort, subject, float(raw.info["sfreq"]), raw.n_times)[0])
    except Exception:                                            # noqa: BLE001
        return 0


# ============================================================================= page rendering
def _page(fig_w_mm: float, fig_h_mm: float):
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    fig.patch.set_facecolor("white")
    return fig


def _header(fig, w_mm, h_mm, blind_id, age, sex, page, total, montage, mains, t0=None,
            sensitivity=True):
    """Identity line left, page line right; settings wrap to a second row on narrow pages."""
    filt = f"{HP} – {LP} Hz" + (f", {mains:.0f} Hz notch" if mains else ", no notch")
    settings = (f"{montage}    {MM_PER_S:.0f} mm/s    {UV_PER_MM:.0f} µV/mm    {filt}"
                if sensitivity else f"{montage}    {filt}")
    # one source cohort records no sex; print nothing rather than a placeholder, which on a
    # clinical page reads as a data error
    ident = f"{blind_id}    age {age}" + (f"    sex {sex}" if str(sex) not in ("?", "", "None")
                                          else "")
    right = f"page {page} of {total}" if t0 is None else f"t = {t0:.0f} s    page {page} of {total}"

    if w_mm >= 250:                       # wide trace page: everything on one line
        fig.text(8 / w_mm, 1 - 7 / h_mm, f"{ident}                {settings}",
                 fontsize=8, family="DejaVu Sans", va="center")
        fig.text(1 - 8 / w_mm, 1 - 7 / h_mm, right, fontsize=8, ha="right", va="center")
    else:
        fig.text(8 / w_mm, 1 - 7 / h_mm, ident, fontsize=8, family="DejaVu Sans", va="center")
        fig.text(1 - 8 / w_mm, 1 - 7 / h_mm, right, fontsize=8, ha="right", va="center")
        fig.text(8 / w_mm, 1 - 13 / h_mm, settings, fontsize=7.2, color="0.3",
                 family="DejaVu Sans", va="center")


def _trace_axes(fig, w_mm, h_mm, n_rows, left_mm=34.0, top_mm=14.0, bottom_mm=16.0):
    """Axes whose data coordinates map exactly to 30 mm/s and 7 uV/mm."""
    ax_w_mm = MM_PER_S * PAGE_S
    row_mm = 10.0
    ax_h_mm = row_mm * n_rows
    ax = fig.add_axes([left_mm / w_mm, (h_mm - top_mm - ax_h_mm) / h_mm,
                       ax_w_mm / w_mm, ax_h_mm / h_mm])
    ax.set_xlim(0, PAGE_S)
    ax.set_ylim(-row_mm * UV_PER_MM * n_rows, 0)      # top row first
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, PAGE_S + 0.001, 1.0))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=3, width=0.6, color="0.35")
    for t in np.arange(1.0, PAGE_S, 1.0):             # 1 s markers, as on a clinical viewer
        ax.axvline(t, color="0.86", lw=0.5, zorder=0)
    return ax, row_mm


def _draw_rows(ax, rows, seg, sf, row_mm, breaks=()):
    t = np.arange(seg) / sf
    step = row_mm * UV_PER_MM
    off = 0.0
    for i, (label, trace) in enumerate(rows):
        if i in breaks:
            off += step * 0.45
        y = -(off + step * 0.5) + trace
        ax.plot(t, y, color="#111111", lw=0.45, solid_joinstyle="miter", antialiased=True)
        ax.text(-0.008, -(off + step * 0.5), label, fontsize=7.2, ha="right", va="center",
                transform=ax.get_yaxis_transform(), family="DejaVu Sans", clip_on=False)
        off += step
    return off


def _calibration(fig, w_mm, h_mm, bottom_mm=16.0, left_mm=34.0):
    """50 uV / 1 s calibration mark, drawn in true page millimetres."""
    x0, y0 = left_mm, bottom_mm - 6.0
    hmm, vmm = MM_PER_S * 1.0, 50.0 / UV_PER_MM
    ax = fig.add_axes([0, 0, 1, 1], zorder=5)
    ax.set_axis_off()
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.plot([x0, x0, x0 + hmm], [y0 + vmm, y0, y0], color="#111111", lw=0.9)
    ax.text(x0 + hmm + 2, y0 + 0.5, "1 s", fontsize=7, va="bottom")
    ax.text(x0 - 1.5, y0 + vmm, "50 µV", fontsize=7, ha="right", va="top")


def bipolar_page(pdf, ch, sf, t0, blind_id, age, sex, page, total, mains):
    rows = []
    a, b = int(t0 * sf), int((t0 + PAGE_S) * sf)
    for hi, lo in BIPOLAR:
        if hi in ch and lo in ch:
            rows.append((f"{hi}–{lo}", ch[hi][a:b] - ch[lo][a:b]))
    n_rows = len(rows) + 0.45 * len([i for i in CHAIN_BREAKS if i < len(rows)]) + 0.6
    w_mm = 34 + MM_PER_S * PAGE_S + 12
    h_mm = 14 + 10 * n_rows + 6 + 16
    fig = _page(w_mm, h_mm)
    _header(fig, w_mm, h_mm, blind_id, age, sex, page, total, "longitudinal bipolar", mains, t0)
    ax, row_mm = _trace_axes(fig, w_mm, h_mm, n_rows)
    _draw_rows(ax, rows, b - a, sf, row_mm, CHAIN_BREAKS)
    _calibration(fig, w_mm, h_mm)
    pdf.savefig(fig)
    plt.close(fig)


def referential_page(pdf, ch, sf, t0, blind_id, age, sex, page, total, mains):
    present = [c for c in REFERENTIAL if c in ch]
    avg = np.mean([ch[c] for c in present], axis=0)
    a, b = int(t0 * sf), int((t0 + PAGE_S) * sf)
    rows = [(f"{c}–avg", ch[c][a:b] - avg[a:b]) for c in present]
    w_mm = 34 + MM_PER_S * PAGE_S + 12
    h_mm = 14 + 10 * len(rows) + 6 + 16
    fig = _page(w_mm, h_mm)
    _header(fig, w_mm, h_mm, blind_id, age, sex, page, total, "referential (average)", mains)
    ax, row_mm = _trace_axes(fig, w_mm, h_mm, len(rows) + 0.6)
    _draw_rows(ax, rows, b - a, sf, row_mm)
    _calibration(fig, w_mm, h_mm)
    pdf.savefig(fig)
    plt.close(fig)


def spectrum_page(pdf, ch, sf, starts, blind_id, age, sex, page, total, mains):
    from scipy.signal import welch
    present = [c for c in POSTERIOR if c in ch]
    avg = np.mean([ch[c] for c in ch], axis=0)
    w_mm, h_mm = 210, 158
    fig = _page(w_mm, h_mm)
    _header(fig, w_mm, h_mm, blind_id, age, sex, page, total,
            "posterior spectrum (average reference)", mains, sensitivity=False)
    ax = fig.add_axes([0.13, 0.15, 0.82, 0.68])
    for c in present:
        # computed over exactly the epochs shown on the trace pages, nothing else
        segs = [ch[c][int(t * sf):int((t + PAGE_S) * sf)] - avg[int(t * sf):int((t + PAGE_S) * sf)]
                for t in starts]
        f, p = welch(np.concatenate(segs), fs=sf, nperseg=int(min(4 * sf, len(segs[0]))))
        m = (f >= 1) & (f <= 30)
        ax.semilogy(f[m], p[m], lw=1.0, label=c)
    ax.set_xlabel("frequency (Hz)", fontsize=8)
    ax.set_ylabel("power spectral density (µV²/Hz)", fontsize=8)
    ax.set_xticks(np.arange(0, 31, 2))
    ax.tick_params(labelsize=7.5)
    ax.grid(True, which="both", color="0.9", lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=len(present))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.13, 0.05, "Unannotated by design: no peak is marked and no frequency is "
                          "reported. Provided only as an adjunct to the trace.",
             fontsize=7, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================= per-recording
def load_raw(cohort: str, subject: str):
    if cohort == "ds004504_raw":
        f = DS / "openneuro_ds004504_raw" / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"
        return mne.io.read_raw_eeglab(str(f), preload=True)
    f = DS / "caueeg-dataset" / "signal" / "edf" / f"{subject}.edf"
    return mne.io.read_raw_edf(str(f), preload=True)


def render_one(rec: dict) -> dict:
    raw = load_raw(rec["cohort"], rec["subject"])
    mains = detect_mains(raw)
    if mains:
        raw.notch_filter(mains, verbose=False)
    raw.filter(HP, min(LP, raw.info["sfreq"] / 2 - 1), verbose=False)

    starts, note = pick_pages(rec["cohort"], rec["subject"], raw)
    ch, sf = as_frame(raw)

    missing = [c for pair in BIPOLAR for c in pair if c not in ch]
    total = len(starts) + 2

    TRACES.mkdir(parents=True, exist_ok=True)
    out = TRACES / f"{rec['blind_id']}.pdf"
    with PdfPages(out) as pdf:
        for p, t0 in enumerate(starts, 1):
            bipolar_page(pdf, ch, sf, t0, rec["blind_id"], rec["age"], rec["sex"],
                         p, total, mains)
        referential_page(pdf, ch, sf, starts[0], rec["blind_id"], rec["age"], rec["sex"],
                         len(starts) + 1, total, mains)
        spectrum_page(pdf, ch, sf, starts, rec["blind_id"], rec["age"], rec["sex"],
                      len(starts) + 2, total, mains)
        pdf.infodict()["Title"] = rec["blind_id"]          # no subject identity in metadata
        pdf.infodict()["Author"] = ""

    seg = np.concatenate([ch[list(ch)[0]][int(t * sf):int((t + PAGE_S) * sf)] for t in starts])
    return {"blind_id": rec["blind_id"], "pages": total, "seconds": len(starts) * PAGE_S,
            "page_starts_s": [round(t, 1) for t in starts], "segment": note,
            "mains": mains, "sfreq": sf,
            "median_abs_uv": round(float(np.median(np.abs(seg))), 2),
            "missing_channels": ",".join(sorted(set(missing))) or "none",
            "file": str(out.relative_to(ROOT))}


# ============================================================================= manifest
def build_manifest(n_per_stratum=(20, 20, 20), seed=20260807) -> list[dict]:
    """Stratified 20 CN / 20 MCI / 20 dementia, blind IDs shuffled so order carries no class."""
    meta = np.load(ROOT / "outputs" / "harmonised_features" / "channel" / "meta.npz",
                   allow_pickle=True)
    g, c, lab = meta["groups"], meta["cohort"], meta["labels"]
    _, first = np.unique(g, return_index=True)
    subs = [(str(g[i]), str(c[i]), str(lab[i])) for i in sorted(first)]

    # demographics, never written to the rendered page beyond age and sex
    part = {r["participant_id"]: r for r in
            csv.DictReader((DS / "openneuro_ds004504_raw" / "participants.tsv").open(),
                           delimiter="\t")}
    cau = {str(r["serial"]): r for r in
           json.loads((DS / "caueeg-dataset" / "annotation.json").read_text())["data"]}

    rng = np.random.default_rng(seed)
    want = {
        "CN":       [("ds004504_raw", {"C"}, 10), ("CAUEEG", {"CN"}, 10)],
        "MCI":      [("CAUEEG", {"MCI"}, 20)],
        "dementia": [("ds004504_raw", {"A"}, 10), ("CAUEEG", {"AD"}, 10)],
    }
    chosen = []
    excluded = []
    for stratum, spec in want.items():
        for cohort, labels, k in spec:
            pool = [s for s in subs if s[1] == cohort and s[2] in labels]
            order = rng.permutation(len(pool))
            # draw in random order, keeping only recordings with enough clean eyes-closed EEG;
            # gradability is a property of the recording, not of the diagnosis, so screening here
            # does not bias the strata
            keep = []
            for i in order:
                if len(keep) == k:
                    break
                gid, coh, lb = pool[i]
                sid = gid.split("_")[-1] if coh == "CAUEEG" else gid
                n = n_clean_pages(coh, sid)
                (keep if n >= MIN_PAGES else excluded).append((gid, coh, lb, sid, n))
            print(f"  {stratum:9s} {cohort:14s} kept {len(keep)}/{k}")
            for gid, coh, lb, sid, n in keep:
                if coh == "ds004504_raw":
                    age, sex = part[sid]["Age"], part[sid]["Gender"]
                else:
                    age, sex = cau.get(sid, {}).get("age", "?"), "?"
                chosen.append({"cohort": coh, "subject": sid, "group_id": gid,
                               "label": lb, "stratum": stratum, "age": age, "sex": sex})

    order = rng.permutation(len(chosen))
    for n, i in enumerate(order, 1):
        chosen[i]["blind_id"] = f"R{n:03d}"
    chosen.sort(key=lambda r: r["blind_id"])

    from collections import Counter
    print(f"  screened out {len(excluded)} recordings with < {MIN_PAGES} clean pages: "
          f"{dict(Counter(x[2] for x in excluded))}")
    return chosen


def write_manifest(recs):
    KEY.parent.mkdir(parents=True, exist_ok=True)
    with KEY.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scoring_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["blind_id", "pdr_frequency_hz", "pdr_well_formed_0_2", "posterior_maximal_yes_no",
                "slowing_none_mild_mod_severe", "slowing_intermittent_continuous",
                "asymmetry_yes_no", "artefact_none_mild_marked", "artefact_location",
                "drowsiness_none_possible_definite",
                "global_impression_normal_borderline_abnormal", "ungradable_yes_no",
                "reader_notes"]
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            w.writerow([r["blind_id"]] + [""] * (len(cols) - 1))
    print(f"sealed key  -> {KEY}   ({len(recs)} recordings)")
    print(f"blank sheet -> {OUT / 'scoring_sheet.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--render", default=None, help="a count, 'all', or R001,R002")
    args = ap.parse_args()

    if args.manifest or not KEY.exists():
        recs = build_manifest()
        write_manifest(recs)
        from collections import Counter
        print("  strata:", dict(Counter(r["stratum"] for r in recs)))
        print("  cohorts:", dict(Counter(r["cohort"] for r in recs)))

    if not args.render:
        return
    recs = list(csv.DictReader(KEY.open(encoding="utf-8")))
    if args.render == "all":
        todo = recs
    elif args.render.startswith("R"):
        ids = set(args.render.split(","))
        todo = [r for r in recs if r["blind_id"] in ids]
    else:
        todo = recs[:int(args.render)]

    report = []
    for i, rec in enumerate(todo, 1):
        try:
            info = render_one(rec)
            report.append(info)
            print(f"  [{i}/{len(todo)}] {info['blind_id']}  {info['pages']} pages  "
                  f"{info['seconds']:.0f} s  median |x| {info['median_abs_uv']} uV  "
                  f"mains {info['mains']}  missing {info['missing_channels']}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {rec['blind_id']}  FAILED: {type(e).__name__}: {e}")
            report.append({"blind_id": rec["blind_id"], "error": f"{type(e).__name__}: {e}"})

    write_result(ROOT / "outputs" / "results" / "audit" / "reader_traces.json",
                 {"rendered": report, "n": len(report)}, script=Path(__file__).name)


if __name__ == "__main__":
    main()
