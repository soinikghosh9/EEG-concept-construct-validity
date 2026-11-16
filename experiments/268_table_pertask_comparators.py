"""Supplementary table: external AUC of the concept model and the eleven comparators, per task.

Reads outputs/results/audit/pertask_benchmark_arm_no_graph.json, written by
    python experiments/178_pertask_benchmark_arm.py --out pertask_benchmark_arm_no_graph.json
and writes the LaTeX table the construct paper's supplement inputs, so the numbers cannot drift.

    python experiments/268_table_pertask_comparators.py
Out: jne_build_revised/tab_pertask_comparators.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "results" / "audit" / "pertask_benchmark_arm_no_graph.json"
OUT = ROOT / "jne_build_revised" / "tab_pertask_comparators.tex"

NAMES = [("concept_model", "Concept model (sign-constrained)"),
         ("random_forest", "Random forest (spectral; reference)"),
         ("gradient_boosting", "Gradient boosting (spectral)"),
         ("SVM", "Support vector machine (spectral)"),
         ("logistic", "Logistic regression (spectral)"),
         ("eegnet", "EEGNetv4"), ("eegitnet", "EEGITNet"), ("shallow", "ShallowFBCSPNet"),
         ("eegnex", "EEGNeX"), ("atcnet", "ATCNet"), ("deep4", "Deep4Net"),
         ("conformer", "EEGConformer")]
TASKS = [("CN_vs_AD", "Alzheimer's"), ("CN_vs_dementia", "Dementia"),
         ("CN_vs_impaired", "Screening"), ("CN_vs_MCI", "Prodromal")]


def main():
    T = json.loads(SRC.read_text())["tasks"]
    dem = T["CN_vs_dementia"]
    rows = []
    for key, label in NAMES:
        cells = [f"{T[t]['auc'][key]:.3f}" if key in T[t]["auc"] else "--" for t, _ in TASKS]
        vr = dem["vs_reference"].get(key)
        if vr is None:
            extra = "--"
        else:
            extra = (f"${vr['delta']:+.3f}$".replace("+-", "-") +
                     f" ({vr['p_holm']:.2f})")
        rows.append(f"{label} & " + " & ".join(cells) + f" & {extra} \\\\")
    sp = dem["spread"]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & Alzheimer's & Dementia & Screening & Prodromal & "
             r"$\Delta$ vs forest, dementia ($p_{\mathrm{Holm}}$) \\", r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}",
             r"\caption{External AUC on each task for the concept model and the eleven comparators "
             r"of the operating-point analysis, all trained directly for each binary contrast under "
             r"one per-task protocol and scored on the same external subjects (Alzheimer's $n=65$, "
             r"dementia and screening $n=87$, prodromal $n=63$). The last column is the paired "
             r"subject-clustered bootstrap difference from the spectral forest on the dementia "
             r"contrast, Holm-adjusted across the family. The observed best-to-worst spread on that "
             rf"contrast, {sp['observed']:.3f}, does not exceed the {sp['null_median']:.3f} median "
             rf"produced by resampling subjects alone ($p={sp['p']:.2f}$). Prodromal values are "
             r"single-seed and should not be read as a ranking.}",
             r"\label{tab:pertask}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
