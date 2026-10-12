# ==============================================================================
# paths.py -- single place that resolves where the data live and where results
# go. Every Python script imports this. Defaults assume the layout documented
# in README.md (data/ and results/ next to this file); override with env vars.
# ==============================================================================
import os
REPO    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.environ.get("GC_DATA_DIR",    os.path.join(REPO, "data"))
RESULTS = os.environ.get("GC_RESULTS_DIR", os.path.join(REPO, "results"))

def data(*parts, env=None):
    """Path under data/, or the file named by the env var `env` if it is set."""
    if env and os.environ.get(env): return os.environ[env]
    return os.path.join(DATA, *parts)

def results(*parts):
    p = os.path.join(RESULTS, *parts); os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p), exist_ok=True); return p

# ---- canonical input files -----------------------------------------------------
MASTER   = data("text", "master", "MASTER_all_models_gender_all_concreteness.csv", env="GC_MASTER_FILE")  # FULL master (rebuilt 5 Oct 2026)
BRYS     = data("norms", "Concreteness_ratings_Brysbaert_et_al_BRM_ext.csv",       env="GC_BRYS_FILE")
LANC     = data("norms", "lancaster_sensorimotor_norms_2019.csv",                   env="GC_LANC_FILE")
IMAGE    = data("images", "image_class_w_conc_PROCESSED.csv",                       env="GC_IMAGE_FILE")
SURPRISE = data("images", "word_surprise_metrics.csv",                              env="GC_SURPRISE_FILE")
SWOW     = data("free_associations", "strength_SWOW-EN_R123_20180827.csv",         env="GC_SWOW_FILE")
LWOW     = data("free_associations", "associations_data_all_scores.json",           env="GC_LWOW_FILE")
AOA      = data("norms", "AoAKuperman.csv",                                         env="GC_AOA_FILE")
VAD      = data("norms", "Ratings_Warriner_et_al.csv",                              env="GC_VAD_FILE")
