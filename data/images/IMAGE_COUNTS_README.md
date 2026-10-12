# Search-term image counts

Exported 9 October 2026 from existing audited artifacts.

- `search_term_image_counts.csv`: `word,image_count` for all 39,918 retained search terms; total 9,977,592 retained image rows.
- `word_surprise_metrics_with_image_counts.csv`: all 38,883 rows and original columns of Doug's `word_surprise_metrics.csv`, with integer `image_count` appended. Every row matched; original values and row order are preserved.

`image_count` counts retained per-image rows in the final object-labeling dataset, not faces, object labels, original downloads, or verified globally unique images. It is suitable as an image-exposure measure for the supplied object metrics. Do not assume it counts the exact images used by the separate face-classification pipeline.

Source: shared artifacts `results/methods_audit/2026-09-17/visual_per_word.csv`, derived from `data/raw/word_to_object_cleaned_22May2025.json`. The existing September audit independently compared populated counts with grouped rows in `intermediate/in_40k_responses_filter_search_word.pkl`, finding no disagreements. Its `crosschecks.json` records the one missing aggregate count, `satellite_TV`, recovered as 30 per-image rows; this export includes that correction.

Matching uses lowercase, underscores replaced by spaces, and collapsed whitespace, with uniqueness assertions. Both `total_objects` and `unique_objects` agree exactly between the count source and every matched metrics row. The standalone lookup preserves source spellings; the augmented spreadsheet preserves Doug's spellings.

The export script remains at `../../../export_image_counts.py` relative to this folder and writes its CSVs here. It reproduces the export and validation from these existing artifacts. No statistical analysis or image processing was rerun. The original spreadsheets were not modified.

## Updated analysis files added 12 October 2026

- `../../figure_04/fig4_image_count_robustness.py`: image-count controls for Figure 4.
- `../../supplementary/si_09_image_face_robustness.py`: face-presence/count robustness for Figure 3.

Both analyses have been verified. Figure 4 uses repository-relative paths, with `GC_SURPRISE_COUNTS_FILE`, `GC_IMAGE_FILE`, and `GC_MASTER_FILE` input overrides, and writes to `figure_04_image_count_robustness/` under the configured results root. SI09 supports `GC_IMAGE_FILE`, `GC_LANC_FILE`, and `GC_RESULTS_DIR`. Input data are available from the authors listed in the repository README.
