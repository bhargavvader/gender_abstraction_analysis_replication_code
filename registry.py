# ==============================================================================
# registry.py -- THE 12-model set used everywhere a text-based gender association
# appears in the paper: Figs. 1, 2, 4c and 5 and Supplementary Sections 3-7.
# Display name -> gender column in MASTER_all_models_gender_all_concreteness.csv
# -> marker (Fig. 2c,d). This is the single source of truth for the model set.
#
# Keep this list in sync with the `registry` tribble in figure_01/figure_01.R.
#
# The master file carries further gender_* columns (gender_GPT2, gender_BERT_2026,
# gender_RoBERTa_2026, gender_word2vec_Retrained, gender_word2vec_2026,
# gender_Twitter_GloVe_200D, gender_FastText_2026) that are not part of the
# model set and are not read by any script.
# ==============================================================================
REGISTRY = [
 ("FastText",        "gender_FastText"),
 ("word2vec",        "gender_word2vec_Original"),
 ("Wikipedia GloVe", "gender_Wikipedia_GloVe"),
 ("ConceptNet",      "gender_ConceptNet"),
 ("BERT",            "gender_BERT"),
 ("distilBERT",      "gender_distilBERT"),
 ("RoBERTa",         "gender_RoBERTa"),
 ("MPNet",           "gender_MPNet"),
 ("GPT4",            "gender_OpenAI_GPT4"),
 ("Qwen3",           "gender_Qwen3_Embedding"),
 ("Nomic v2 MoE",    "gender_Nomic_v2_MoE"),
 ("GPT2",            "gender_GPT2_2026"),
]
SHAPES = {"FastText": "p", "word2vec": "s", "Wikipedia GloVe": "^", "ConceptNet": "h", "BERT": "<", "distilBERT": ">",
          "RoBERTa": "d", "MPNet": "H", "GPT4": "*", "Qwen3": "X", "Nomic v2 MoE": "P", "GPT2": "v"}
