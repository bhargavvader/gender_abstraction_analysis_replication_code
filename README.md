# The Male Abstraction Bias in Internet Culture and Large Language Models — reproduction code

Scripts that regenerate every main figure (Figs. 1–5) and every supplementary analysis
(Supplementary Sections 2–8, Tables 2–11, Supplementary Fig. 1) of the manuscript. Each script is the
final version used for the paper; only the path/CONFIG lines were changed so that everything resolves
relative to this repository.

**One model set, everywhere.** Every text-based gender association in the paper — Figs. 1, 2, 4c and 5
and Supplementary Sections 3–7 — uses the same 12 language models, defined once in `registry.py`
(FastText, word2vec, Wikipedia GloVe, ConceptNet, BERT, distilBERT, RoBERTa, MPNet, GPT4, Qwen3,
Nomic v2 MoE, GPT2), read from the gender columns of `MASTER_all_models_gender_all_concreteness.csv`.

## Layout

```
.
├── README.md
├── requirements.txt
├── registry.py               THE 12-model set (display name -> master column -> Fig. 2 marker); single source of truth
├── paths.py                  resolves data/ and results/ (env-var overridable); imported by every Python script
├── data/                     supplied CSV summaries and locations for additional inputs (see "Data" below)
├── results/                  all outputs are written here, one sub-folder per script
├── figure_01/
│   ├── common.py                 loader + panel definitions (Python Fig. 1)
│   ├── figure_01_compute.py      decile means, inset Ns, per-model r, across-model summaries  -> results/figure_01/
│   ├── figure_01_plot.py         renders fig1_text_data.png/.pdf (this produced the manuscript file)
│   └── figure_01.R               ggplot2/patchwork equivalent (same registry, same geometry)
├── figure_02/
│   ├── figure_02_a_domains_flags_classes.py   WordNet domains, exclusion flags, word forms -> vocab_L3_classified.csv, all_terms_domains_flags.csv
│   ├── figure_02_b_terciles_centroids.py      within-domain terciles + construal centroids    -> fig2_*.csv
│   └── figure_02_c_figure.py                  renders fig2_by_domain.png/.pdf
├── figure_03/
│   └── figure_03.R               two-panel P(Male Face) figure -> FIG_two_panel_image_pmale_all-terms.png (= fig3_image_results)
├── figure_04/
│   └── figure_04.py              unique objects vs concreteness / image gender / text gender (12-model composite) -> fig4_men_are_many.png
├── figure_05/
│   ├── figure_05_compute.py      SWOW + LWOW free associations x the 12 registry scorers -> master12_correlations.csv
│   └── figure_05_plot.py         stacked line plot + heat map from that table -> fig5_free_association.png
└── supplementary/
    ├── domain_labels_40k_wordnet_POSfiltered.csv   TRACKED: POS-filtered WordNet domain map for SI 6 (frozen; see "Data")
    ├── si_02_image_face_count_robustness.R    Supp. Section 2 / Table 2   (17 specifications x 2 predictors)
    ├── si_03_04_fig1_supp_tables.py           Supp. Sections 3-4 / Tables 3-4 (LaTeX from the Fig. 1 statistics)
    ├── si_05_lexical_covariates.py            Supp. Section 5 / Tables 5-8 (covariate, reliability, composition, leave-one-domain-out)
    ├── si_06_word_level_examples.py           Supp. Section 6 / Supp. Fig. 1 / Table 9 (12 models, 48-cell FDR family)
    ├── si_07_domain_assignment_sensitivity.py Supp. Section 7 / Table 10 (primary / all-senses / unambiguous / 200 random replicates)
    ├── si_08_image_regressions.py             Supp. Section 8 / Table 11 (+ all Fig. 3 statistics in the main text)
    └── si_make_tex_tables.py                  writes stats_fig1.tex and stats_si_text.tex (the files the manuscript \input's)
```

## Data

The supplied CSV summaries are included in this repository. Remaining input data are available on request. For the data files used in these analyses, contact:

- Douglas Guilbeaut: [dguilb@stanford.edu](mailto:dguilb@stanford.edu)
- Bhargav Srinivasa Desikan: [bhargav.srinivasadesikan@oii.ox.ac.uk](mailto:bhargav.srinivasadesikan@oii.ox.ac.uk)

Put the input files at these paths (or point the `GC_*` environment variables listed in `paths.py`
at them). CSV summaries and documentation under `data/` are tracked; the full raw data are not included.

| Path | File | Used by | Where to get it |
|---|---|---|---|
| `data/text/master/` | `MASTER_all_models_gender_all_concreteness.csv` — the **full** master (gender columns for all 12 registry models incl. Wikipedia GloVe word scores; 106,385 rows × 43 cols) | Fig 1, Fig 2, Fig 4, Fig 5, SI 5, SI 6, SI 7 | Contact the authors; request the full master, not the `_SIMPLE` copy. |
| `data/norms/` | `Concreteness_ratings_Brysbaert_et_al_BRM_ext.csv` (Word, Conc.M, Conc.SD, Percent_known, SUBTLEX, Dom_Pos) | Fig 2, Fig 4, SI 5, SI 6, SI 7 | Contact the authors |
| `data/norms/` | `lancaster_sensorimotor_norms_2019.csv` | Fig 3, SI 2, SI 8 | Contact the authors |
| `data/norms/` | `AoAKuperman.csv` (Kuperman et al. 2012 AoA, 51,715 words; columns `Word`, `AoA_Kup`, `none` = missing) | SI 5 | Kuperman et al. (2012) release; the copy used is the one bundled in the `lingfeat` PyPI package |
| `data/norms/` | `Ratings_Warriner_et_al.csv` (Warriner et al. 2013 VAD; `Word`, `V.Mean.Sum`, `A.Mean.Sum`, `D.Mean.Sum`; first column a row index) | SI 5 | Warriner et al. (2013) release |
| `data/images/` | `image_class_w_conc_PROCESSED.csv` (one row per search term: num_faces, num_males, frac_male, conc, freq, gender_conf, Dom_Pos, …) | Fig 3, Fig 4, SI 2, SI 8, SI 9 | Contact the authors |
| `data/images/` | `word_surprise_metrics.csv` (visual-divergence interface v1; 38,883 rows × 16 cols; `unique_objects`) | Fig 4 | Contact the authors |
| `data/images/` | `word_surprise_metrics_with_image_counts.csv` (metrics with retained image counts appended) | Fig 4 image-count robustness | Included |
| `data/images/` | `search_term_image_counts.csv` (standalone retained image counts by search term) | Supporting data | Included |
| `data/free_associations/` | `strength_SWOW-EN_R123_20180827.csv` (SWOW strength release; tab-separated with unescaped quotes — read with `quoting=csv.QUOTE_NONE`; 1,389,711 cue–response pairs, 12,216 cues) | Fig 5 | Small World of Words (De Deyne et al. 2019) |
| `data/free_associations/` | `associations_data_all_scores.json` (LWOW processed release) | Fig 5 | LLM World of Words (Abramski et al. 2025) |

Tracked derived input: `supplementary/domain_labels_40k_wordnet_POSfiltered.csv` is the POS-filtered WordNet
domain map used by SI 6 (12,705 word–domain rows: verb.communication 1,393, adj.all 5,024, noun.artifact 4,634,
adv.all 1,654). It is committed so that the analysis does not drift with WordNet releases; if the file is deleted,
`si_06_word_level_examples.py` rebuilds it from WordNet with the same rule (note: `pandas` reads the word *null* as
missing, so a rebuild from a default-parsed Brysbaert file omits it, matching the committed map).

## Dependencies

Python ≥ 3.10: `pip install -r requirements.txt`, then
`python -c "import nltk; [nltk.download(r) for r in ('wordnet','names','averaged_perceptron_tagger_eng','cmudict')]"`.
SciPy ≥ 1.11 is required by SI 6 (`false_discovery_control`).

R ≥ 4.2 with `dplyr`, `tidyr`, `readr`, `tibble`, `ggplot2`, `patchwork` (≥ 1.2.0), `scales`, `grid`,
`sandwich`, `lmtest`. The R scripts locate the repo from their own path; run them with
`Rscript figure_01/figure_01.R` etc. (or set `GC_REPO_DIR`).

## Run order

Python scripts can be run from anywhere (they resolve paths from their own location). Arrows mark
dependencies on earlier outputs.

```
# Fig. 1 (text results)                                  -> results/figure_01/
python figure_01/figure_01_compute.py
python figure_01/figure_01_plot.py            # or: Rscript figure_01/figure_01.R

# Fig. 2 (within-domain + word form)                     -> results/figure_02/
python figure_02/figure_02_a_domains_flags_classes.py      # ~10-15 min (WordNet lookups for 106k terms)
python figure_02/figure_02_b_terciles_centroids.py
python figure_02/figure_02_c_figure.py

# Fig. 3 (Google-image faces)                            -> results/figure_03/
Rscript figure_03/figure_03.R                 # GC_WORD_SET=faces for the >=1-face variant

# Fig. 4 (men are many)                                  -> results/figure_04/
python figure_04/figure_04.py

# Fig. 5 (free associations)                             -> results/figure_05/
python figure_05/figure_05_compute.py                      # ~3 min
python figure_05/figure_05_plot.py

# Supplementary
Rscript supplementary/si_02_image_face_count_robustness.R          -> results/si_02/   (Table 2)
python  supplementary/si_03_04_fig1_supp_tables.py                 -> results/si_03_04/ (Tables 3-4; needs results/figure_01/)
python  supplementary/si_05_lexical_covariates.py                  -> results/si_05/   (Tables 5-8; needs results/figure_02/all_terms_domains_flags.csv)
python  supplementary/si_06_word_level_examples.py                 -> results/si_06/   (Supp. Fig. 1, Table 9)
python  supplementary/si_07_domain_assignment_sensitivity.py       -> results/si_07/   (Table 10; ~20 min for the 200 replicates)
python  supplementary/si_08_image_regressions.py                   -> results/si_08/   (Table 11 + Fig. 3 statistics)
python  supplementary/si_make_tex_tables.py                        -> results/tex/     (stats_fig1.tex, stats_si_text.tex; needs figure_01 + si_05)
```

Manuscript figure files: `results/figure_01/fig1_text_data.*`, `results/figure_02/fig2_by_domain.*`,
`results/figure_03/FIG_two_panel_image_pmale_all-terms.*` (= `fig3_image_results`),
`results/figure_04/fig4_men_are_many.*`, `results/figure_05/fig5_free_association.*`,
`results/si_06/figS_domain_word_examples.png` (= Supplementary Fig. 1).

## Mapping from manuscript to scripts

| Manuscript | Script | Notes |
|---|---|---|
| Fig. 1 + first results paragraphs; Supp. Sections 3–4 | `figure_01_compute.py` (+ `si_make_tex_tables.py` for the `\FigOne*` macros) | computes all scopes: pooled / words / phrases for human concreteness and for Lancaster strength |
| Fig. 2 + domain and word-form paragraphs | `figure_02_*` | vocabulary reproduces exactly: 31,591 words / 35 domains; 56,879 phrases / 33 domains |
| Fig. 3 | `figure_03.R`; statistics in `si_08_image_regressions.py` | main figure = all terms, zero-face terms coded 0 |
| Fig. 4 | `figure_04.py` | panel c: equal-weight mean of the 12 registry z-scores; r = 0.07 [0.06, 0.08], ρ = 0.10 |
| Fig. 5 | `figure_05_compute.py` → `figure_05_plot.py` | 12 registry scorers; n = 11,179 SWOW cues (all scorers), 10,495–10,527 LWOW cues; across-model t-tests in the text are computed from the 5×12 table |
| Supp. Table 2 | `si_02_image_face_count_robustness.R` | standardized β, HC1 |
| Supp. Tables 3–4 | `si_03_04_fig1_supp_tables.py` / `si_make_tex_tables.py` | |
| Supp. Tables 5–8 | `si_05_lexical_covariates.py` → `si_make_tex_tables.py` | |
| Supp. Fig. 1, Table 9 | `si_06_word_level_examples.py` | 12 models × 4 domains = 48-cell FDR family; row label GPT4 |
| Supp. Table 10 | `si_07_domain_assignment_sensitivity.py` | seed 2026 |
| Supp. Table 11 | `si_08_image_regressions.py` | unstandardized β, HC1; quasi-binomial in log-odds |

## Verification status

All analysis scripts have been verified against the analysis data, including
`figure_04/fig4_image_count_robustness.py` and `supplementary/si_09_image_face_robustness.py`.

The additional robustness analyses can be run with:

```bash
python figure_04/fig4_image_count_robustness.py
python supplementary/si_09_image_face_robustness.py
```

The Figure 4 robustness script reads `data/images/word_surprise_metrics_with_image_counts.csv`
(override with `GC_SURPRISE_COUNTS_FILE`), uses `GC_IMAGE_FILE` and `GC_MASTER_FILE` for optional
input-path overrides, and writes to `results/figure_04_image_count_robustness/` under the configured
results root. If the master is absent, its text-gender panel uses a clearly labelled provisional
single-model fallback. SI 9 supports `GC_IMAGE_FILE`, `GC_LANC_FILE`, and `GC_RESULTS_DIR` overrides
and defaults to `results/si_09/`.

## Not included

The gender-scoring pipelines that produced the `gender_*` columns of the master (embedding models,
geometry-of-culture axis) and the Llama-3.2-vision image classification live in the images-objects repository and
are not duplicated here. The LaTeX statistics files the manuscript \input's (`stats_fig1.tex`, `stats_si_text.tex`) are regenerated by
`si_make_tex_tables.py`; no manuscript-editing scripts are included.
