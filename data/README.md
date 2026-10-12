# data/

Added 12 October 2026: `images/search_term_image_counts.csv` and
`images/word_surprise_metrics_with_image_counts.csv`. See
[image-count provenance and updated-script locations](images/IMAGE_COUNTS_README.md).

Input files are not tracked. Place them exactly here (or set the GC_* environment variables in ../paths.py):

    data/text/master/MASTER_all_models_gender_all_concreteness.csv     full master (NOT the _SIMPLE copy)
    data/norms/Concreteness_ratings_Brysbaert_et_al_BRM_ext.csv
    data/norms/lancaster_sensorimotor_norms_2019.csv
    data/norms/AoAKuperman.csv                                          Kuperman et al. 2012 (SI Section 5 only)
    data/norms/Ratings_Warriner_et_al.csv                               Warriner et al. 2013 (SI Section 5 only)
    data/images/image_class_w_conc_PROCESSED.csv
    data/images/word_surprise_metrics.csv                               visual-divergence interface v1 (Fig. 4 only)
    data/free_associations/strength_SWOW-EN_R123_20180827.csv           SWOW strength release (Fig. 5 only)
    data/free_associations/associations_data_all_scores.json            LWOW (Fig. 5 only)

The POS-filtered WordNet domain map used by SI 6 is tracked at ../supplementary/domain_labels_40k_wordnet_POSfiltered.csv.
See ../README.md for the full table (sources, which script uses what).
