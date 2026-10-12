# ==============================================================================
# figure_01_compute.py  (= fig1_stats_12models.py, 5 Oct 2026)
#
# Computes everything behind Fig. 1 and Supplementary Sections 3-4:
#   * per-model decile means of the plotted measure (dplyr::ntile deciles of
#     each model's gender score)                  -> fig1_text_data_decile_means.csv
#   * inset Ns (terms with the measure AND >= 1 plotted model score) -> fig1_Ns.json
#   * per-model Pearson r, Fisher 95% CI, Spearman rho, Bonferroni flag, decile
#     1 / 10 means, for every panel x scope        -> fig1_per_model_correlations.csv
#       A: human concreteness   scopes All / Wrd / Phr
#       B: GPT concreteness     scope  Phr
#       C: Lancaster strength   scopes Wrd / Phr / All   (Phr/All = SI extension)
#   * across-model summaries, models as units, equal weight: mean r, 95% t CI,
#     range, one-sample t, counts                   -> fig1_across_model_summary.csv
#
# Input : the FULL master (19 gender_* columns). Output dir: results/figure_01/.
# ==============================================================================
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, json
from scipy import stats
import common as c
from paths import results
OUT = results('figure_01')
m = c.load()
MIN_TERMS = 100
dec_rows, stat_rows, Ns = [], [], {}

def pearson_block(x, y):
    n = len(x); r, p = stats.pearsonr(x, y)
    z = np.arctanh(r); se = 1/np.sqrt(n-3)
    lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    rho = stats.spearmanr(x, y).correlation
    return dict(n=n, r=r, ci_lo=lo, ci_hi=hi, p=p, rho=rho)

# scopes for stats: panel letter + scope label
SCOPES = [('A','All',['word','phrase'],'conc_brysbaert'), ('A','Wrd',['word'],'conc_brysbaert'), ('A','Phr',['phrase'],'conc_brysbaert'),
          ('B','Phr',['phrase'],'conc_gpt_prob'),
          ('C','Wrd',['word'],'lanc_mean_perceptual'), ('C','Phr',['phrase'],'lanc_mean_perceptual'), ('C','All',['word','phrase'],'lanc_mean_perceptual')]
for panel, scope, dsets, measure in SCOPES:
    d = m[m.dataset.isin(dsets) & m[measure].notna()]
    used_cols = []
    for name, col in c.REGISTRY:
        x = d[[col, measure, 'dataset']].dropna(subset=[col])
        if len(x) < MIN_TERMS:
            print(f'  {panel}/{scope}: skipping {name} ({len(x)} terms)'); continue
        used_cols.append(col)
        dec = c.ntile(x[col], 10)
        g = x.groupby(dec)[measure].mean()
        s = pearson_block(x[col].values, x[measure].values)
        s.update(panel=panel, scope=scope, model=name, master_col=col, n_word=int((x.dataset=='word').sum()), n_phrase=int((x.dataset=='phrase').sum()),
                 dec1=g.loc[1], dec10=g.loc[10])
        stat_rows.append(s)
        # figure decile means only for the plotted scope of each panel
        if dsets == c.PANELS[panel]['scope']:
            for k, v in g.items():
                dec_rows.append(dict(panel=panel, model=name, decile=int(k), value=v, n_terms=int((dec==k).sum())))
    if dsets == c.PANELS[panel]['scope']:
        scored = d[used_cols].notna().any(axis=1)
        Ns[panel] = {ds: int(((d.dataset==ds) & scored).sum()) for ds in dsets}

dec = pd.DataFrame(dec_rows); st = pd.DataFrame(stat_rows)
# Bonferroni within each panel/scope column
st['k_models'] = st.groupby(['panel','scope'])['model'].transform('count')
st['p_bonf'] = np.minimum(st.p * st.k_models, 1.0)
st['bonf_sig'] = st.p_bonf < 0.05
dec.to_csv(os.path.join(OUT, 'fig1_text_data_decile_means.csv'), index=False)
st.to_csv(os.path.join(OUT, 'fig1_per_model_correlations.csv'), index=False)
json.dump(Ns, open(os.path.join(OUT, 'fig1_Ns.json'),'w'), indent=1)
print('Ns:', Ns)

# across-model summaries (models as units, equal weight)
summ = []
for (panel, scope), t in st.groupby(['panel','scope'], sort=False):
    rr = t.r.values; k = len(rr); mu = rr.mean(); se = rr.std(ddof=1)/np.sqrt(k); tcrit = stats.t.ppf(0.975, k-1)
    tt = stats.ttest_1samp(rr, 0)
    summ.append(dict(panel=panel, scope=scope, k=k, mean_r=mu, ci_lo=mu-tcrit*se, ci_hi=mu+tcrit*se, min_r=rr.min(), max_r=rr.max(),
                     t=tt.statistic, df=k-1, p=tt.pvalue, k_neg=int((rr<0).sum()), k_sig_neg=int(((rr<0)&t.bonf_sig).sum()), k_sig_pos=int(((rr>0)&t.bonf_sig).sum()),
                     dec1_mean=t.dec1.mean(), dec10_mean=t.dec10.mean(), n_min=int(t.n.min()), n_max=int(t.n.max())))
summ = pd.DataFrame(summ); summ.to_csv(os.path.join(OUT, 'fig1_across_model_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(summ.round(3).to_string(index=False))
print()
print(st[['panel','scope','model','n','r','ci_lo','ci_hi','p','bonf_sig','dec1','dec10']].round(3).to_string(index=False))
