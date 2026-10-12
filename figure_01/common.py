# ==============================================================================
# common.py -- shared loader / registry / panel definitions for Fig. 1 (Python)
# Recovered from the final 12-model run of 5 Oct 2026 (fig1_stats_12models.py /
# fig1_plot_12models.py). Only the path lines differ from the original.
# ==============================================================================
import os, sys
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MASTER            # rebuilt master (5 Oct 2026) with Wikipedia GloVe word scores
from registry import REGISTRY       # 12-model set shared with Fig. 2; legend order = this order

PANELS = {
 'A': dict(title="Human Judgments\nof Concreteness",          measure='conc_brysbaert',       scope=['word','phrase'], ylab="Mean Concreteness Rating\n(1–5)"),
 'B': dict(title="AI Judgments\nof Concreteness",             measure='conc_gpt_prob',        scope=['phrase'],        ylab="Mean Concreteness Rating\n(GPT Norms, 1–5)"),
 'C': dict(title="Human Judgments of\nSensorimotor Strength", measure='lanc_mean_perceptual', scope=['word'],          ylab="Mean Perceptual Strength\n(Lancaster Norms)"),
}
def load():
    m = pd.read_csv(MASTER, low_memory=False, keep_default_na=False, na_values=['','NA','NaN'])
    for c in ['conc_brysbaert','conc_gpt_prob','lanc_mean_perceptual'] + [c for _,c in REGISTRY]:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m['key'] = m['term'].astype(str).str.strip().str.lower()
    return m
def ntile(x, n=10):
    # dplyr::ntile: equal-size bins by rank, ties by first appearance
    r = x.rank(method='first')
    return np.floor((r - 1) * n / len(x)).astype(int) + 1
