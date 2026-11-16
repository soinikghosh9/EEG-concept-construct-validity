# Construct validity of interpretable EEG concepts for dementia triage

Code to reproduce the analyses in

> S. Ghosh, R. N. Chaurasia, A. K. Maurya, M. Srivastava, S. Sharma and N. Sharma, "Construct validity
> of interpretable EEG concepts for dementia triage: a measurement battery, blinded expert reading and
> multi-cohort external validation," manuscript submitted to *Journal of Neural Engineering*, 2026.

The study asks whether the named clinical concepts an interpretable resting-EEG triage model reports
(posterior dominant rhythm frequency, slowing ratios, posterior maximality, asymmetry and so on)
measure what their names denote. It combines a seven-check construct-validation battery, a blinded
reading of 60 recordings by two neurologists, and external evaluation at clinics absent from
training against two free baselines: patient age and a spectral random forest.

This repository holds code and the blinded reader grades only. It contains no EEG data, no
participant-level labels and no trained weights.

## Data

| Cohort | Role | Access |
|---|---|---|
| CAUEEG | training (all four classes) | by application, under its data-use agreement |
| OpenNeuro ds004504 | external: Alzheimer's and dementia contrasts | open (OpenNeuro) |
| GENEEG | external: prodromal contrast | public release |
| MISP, P-ADIC | single-class; distributional analyses only | public releases |
| EasyCog | wearable montage; construct analyses only | public release, github.com/EasyCog/EasyCog-Benchmark |

Accession details and citations are in the paper's cohort table. Paths are set in
`src/config/config.py`.

## Environment

Python 3.12. `pip install -r requirements.txt`. The reported runs used PyTorch 2.11 with CUDA 12.8 on
one 8 GB GPU; only the deep comparators (scripts 131, 161) need a GPU.

## Pipeline

Run from the repository root. Each script writes one named artefact under `outputs/` with a
provenance block (commit, library versions, arguments), and figures and tables are drawn from those
artefacts rather than recomputed.

**1. Corpus and concepts**

    python experiments/124_fetch_raw_ds004504.py
    python experiments/126_harmonise_corpus.py          # acquisition matching: length, epoch quota, artefact burden
    # the reported arms (exact arguments from the artefacts' provenance blocks)
    python experiments/127_extract_harmonised.py --mode channel --tag channel_clean \
        --cohorts CAUEEG,ds004504_raw,GENEEG,MCI_Dataset,AlzEEG,MISP,P-ADIC \
        --truncate CAUEEG:235,ds004504_raw:235,GENEEG:235,MISP:235,P-ADIC:235,MCI_Dataset:50 \
        --fixed-windows 24
    python experiments/127_extract_harmonised.py --mode global --tag global_clean \
        --cohorts CAUEEG,ds004504_raw,GENEEG,MCI_Dataset,AlzEEG,MISP,P-ADIC \
        --truncate CAUEEG:235,ds004504_raw:235,GENEEG:235,MISP:235,P-ADIC:235,MCI_Dataset:50 \
        --fixed-windows 24

MCI_Dataset and AlzEEG are extracted but excluded from analysis (screening criteria in the
supplement's participant flow).

**2. Construct-validation battery**

| Script | Check |
|---|---|
| 129_concept_validity_harmonised | criterion, convergent, discriminant and reliability verdicts per concept |
| 171_convergent_validity_matched | convergent validity on the matched corpus |
| 117_artefact_proxy_validation, 172_artefact_instrument_sources, 176_instrument_band_composition, 125_artefact_composition, 123_ica_muscle_check | what the artefact instrument measures |
| 143_emg_sweep | analysis-band ceiling sweep |
| 121_topographic_construct, 148_amplitude_concepts | responsiveness of topographic concepts; amplitude-referenced forms |
| 174_discriminant_effect_sizes, 168_acquisition_floors, 167_contribution_harmonised | effect sizes, acquisition floors, contribution share |

**3. Models and external evaluation**

| Script | Result |
|---|---|
| 157_task_battery | sign-constrained concept model, per task; external AUCs |
| 128_benchmark_harmonised, 134_paired_tests, 153_benchmark_tost | spectral forest and classical models (three-class arm); paired tests; equivalence tests |
| 165_benchmark_sign_corrected, 139_sensitivity | sign-corrected concept model; regularisation, aggregation and band-ceiling sensitivity |
| 161_deep_per_task, 131_deep_benchmark | seven published deep architectures (braindecode reference implementations) |
| 178_pertask_benchmark_arm | paired subject-clustered bootstrap against the forest, Holm-adjusted; spread null. Run with `--out pertask_benchmark_arm_no_graph.json` to produce the file the table and figure scripts read |
| 169_cost_decomposition_harmonised (uses 119), 133_binary_heads, 138_concept_ablation | cost of the sign constraints, by estimand |
| 154_repaired_features_benchmark | amplitude-referenced substitution in the two-stage instrument |
| 163_concept_task_profile, 164_concept_sign_audit | which concepts carry which task |
| 155_threshold_sensitivity, 137_deployment_endpoints, 136_reliability_harmonised | operating point, calibration, split-half reliability |
| 173_age_confound_and_earlyonset, 196_increment_over_age, 197_official_split_leakage, 198_net_benefit_prevalence | age confound, increment over age, leakage check, net benefit |
| 175_prodromal_decomposition, 177_sex_confound, 151_ad_vs_ftd, 158_single_class_abnormality | prodromal contrast, sex, AD vs FTD, single-class cohorts |
| 190_wearable_montage_ablation, 191_easycog_battery, 192_core_panel | forehead-and-ear montage |

**4. Blinded expert reading**

    python experiments/146_export_reader_traces.py --manifest      # seeded sample of 60 recordings, sealed key
    python experiments/146_export_reader_traces.py --render all    # the reading packet and a blank sheet
    python experiments/266_reader_study_analysis.py             # agreement, estimator vs readers, sensitivity analyses

The packet and the sealed key are regenerated from the original accessions and are not distributed,
because they derive from CAUEEG recordings. `reader_study/` holds the two readers' grades keyed by
blinded recording ID (`reader1_grades.csv`, `reader2_grades.csv`) and the scoring codebook
(`scoring_codebook.csv`); the grades cannot be linked to diagnoses without the regenerated key.

**5. Figures and tables**

    python experiments/166_figures_v2.py            # figure 1 (design), concept-importance figure
    python experiments/144_manuscript_figures.py    # report, validity, failure-mode, composition figures
    python experiments/159_clinical_view.py         # clinical view
    python experiments/194_fig_operating.py         # operating-point figure
    python experiments/267_fig_reader_study.py      # reader-study figure
    python experiments/145_supplementary_figures.py
    python experiments/156_flow_diagram.py
    python experiments/268_table_pertask_comparators.py

`experiments/sciplots.py` and `experiments/figtools.py` are the figure toolkit used by the reader
figure.

## Notes

- Environment variables carry the legacy prefix `NCG_` (for example `NCG_NORM`); they are
  configuration names only.
- The stored per-subject predictions for the prodromal task come from a later seed sweep than the
  published per-task table, so script 178 skips paired tests on that task and says so.

## License

MIT; see `LICENSE`.
