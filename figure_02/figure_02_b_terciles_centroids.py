"""figure_02_b_terciles_centroids.py  (= fig2_by_domain/02_terciles_centroids.py, 5 Oct 2026)

Fig. 2 pipeline, step 2 (same computations as the Sept 2026 scripts 09_terciles_across_models.py and
08_construal_by_model.py, on the 12-model registry):
  * Within-domain tercile contrast: for each WordNet domain with >= 150 terms in the Fig. 2 vocabulary, tercile
    cut-points on the domain's concreteness; for each model (gender z-scored on the vocabulary it covers; >= 20 terms per
    tercile) the mean z of the most abstract and most concrete thirds; equal-weight mean across models with 95% CI
    (t distribution, models as units); paired t-test across models; binomial sign test over domains.
  * Construal-class centroids: per model, mean z-scored gender and concreteness per word form; equal-weight mean across
    models with 95% CI; class differences tested with models as units (Wilcoxon signed-rank).
Reads results/figure_02/vocab_L3_classified.csv; writes results/figure_02/fig2_*.csv.
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from paths import results
from registry import REGISTRY
OUT = results("figure_02")
CONC = "conc_brysbaert"; MIN_DOMAIN = 150; MIN_TERCILE = 20; MIN_MODEL_COVERAGE = 1000
v = pd.read_csv(os.path.join(OUT, "vocab_L3_classified.csv"), low_memory=False)
MODELS = [(n, c) for n, c in REGISTRY]
def z(x): return (x - x.mean()) / x.std(ddof=0)
def tci(x):
    n = len(x); return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

# ------------------------------------------------------------------ terciles --------------------------------
def terciles(d):
    models = [(n, c) for n, c in MODELS if d[c].notna().sum() > MIN_MODEL_COVERAGE]
    zg = {n: z(d[c].dropna()) for n, c in models}
    rows, per_model_rows = [], []
    for dom, g in d.dropna(subset=["wn_lexname"]).groupby("wn_lexname"):
        if len(g) < MIN_DOMAIN: continue
        lo, hi = g[CONC].quantile([1/3, 2/3])
        abs_idx, con_idx = g.index[g[CONC] <= lo], g.index[g[CONC] >= hi]
        pm = []
        for n, c in models:
            a = zg[n].reindex(abs_idx).dropna(); k = zg[n].reindex(con_idx).dropna()
            if len(a) < MIN_TERCILE or len(k) < MIN_TERCILE: continue
            pm.append((n, a.mean(), k.mean())); per_model_rows.append(dict(domain=dom, model=n, abstract=a.mean(), concrete=k.mean(), n_abs=len(a), n_con=len(k)))
        pm = pd.DataFrame(pm, columns=["model", "abstract", "concrete"]); diff = pm.abstract - pm.concrete
        rows.append(dict(domain=dom, N_terms=len(g), n_models=len(pm), cls=g.cls.mode().iloc[0] if g.cls.notna().any() else np.nan,
                         domain_mean_conc=g[CONC].mean(),
                         abstract_mean=pm.abstract.mean(), abstract_ci=tci(pm.abstract),
                         concrete_mean=pm.concrete.mean(), concrete_ci=tci(pm.concrete),
                         diff_mean=diff.mean(), diff_ci=tci(diff), paired_t_p=stats.ttest_rel(pm.abstract, pm.concrete).pvalue,
                         n_models_abstract_more_male=int((diff > 0).sum())))
    return pd.DataFrame(rows).sort_values("diff_mean", ascending=False), pd.DataFrame(per_model_rows)

# ------------------------------------------------------------------ centroids -------------------------------
WORD_ORDER = ["Adverb", "Verb", "Adjective", "Noun: event/state", "Noun: entity"]
PHR_ORDER = ["Modifier phrase", "Verb phrase / clause", "Noun phrase: event/state", "Noun phrase: entity"]
def centroids(d, order):
    rows = []
    for n, c in MODELS:
        x = d[[c, CONC, "cls"]].dropna(subset=[c, "cls"]).copy()
        if len(x) < MIN_MODEL_COVERAGE: continue
        x["zg"], x["zc"] = z(x[c]), z(x[CONC])
        for cl in order:
            g = x[x.cls == cl]
            rows.append(dict(model=n, cls=cl, N=len(g), z_gender=g.zg.mean(), z_conc=g.zc.mean(), conc_raw=g[CONC].mean()))
    out = pd.DataFrame(rows); mean_rows = []
    for cl in order:
        g = out[out.cls == cl]
        mean_rows.append(dict(model="mean_across_models", cls=cl, N=int(d[d.cls == cl].shape[0]), n_models=len(g),
                              z_gender=g.z_gender.mean(), ci_gender=tci(g.z_gender), z_conc=g.z_conc.mean(), ci_conc=tci(g.z_conc),
                              conc_raw=d[d.cls == cl][CONC].mean()))
    return pd.concat([out, pd.DataFrame(mean_rows)], ignore_index=True)
def class_contrasts(c, pairs):
    rows = []
    m = c[c.model != "mean_across_models"]
    for a, b in pairs:
        pa = m[m.cls == a].set_index("model").z_gender; pb = m[m.cls == b].set_index("model").z_gender
        d = (pa - pb).dropna()
        rows.append(dict(contrast=f"{a} - {b}", n_models=len(d), diff=d.mean(), ci=tci(d), n_positive=int((d > 0).sum()),
                         wilcoxon_p=stats.wilcoxon(d).pvalue))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    w, p = v[v.dataset == "word"], v[v.dataset == "phrase"]
    tw, tw_pm = terciles(w); tp, tp_pm = terciles(p)
    tw.to_csv(f"{OUT}/fig2_terciles_words.csv", index=False); tp.to_csv(f"{OUT}/fig2_terciles_phrases.csv", index=False)
    tw_pm.to_csv(f"{OUT}/fig2_terciles_per_model_words.csv", index=False); tp_pm.to_csv(f"{OUT}/fig2_terciles_per_model_phrases.csv", index=False)
    cw, cp = centroids(w, WORD_ORDER), centroids(p, PHR_ORDER)
    cw.to_csv(f"{OUT}/fig2_centroids_words.csv", index=False); cp.to_csv(f"{OUT}/fig2_centroids_phrases.csv", index=False)
    kw = class_contrasts(cw, [("Verb", "Noun: entity"), ("Adverb", "Noun: entity"), ("Adjective", "Noun: entity"), ("Noun: event/state", "Noun: entity")])
    kp = class_contrasts(cp, [("Verb phrase / clause", "Noun phrase: entity"), ("Modifier phrase", "Noun phrase: entity"), ("Noun phrase: event/state", "Noun phrase: entity")])
    kw.assign(dataset="word").to_csv(f"{OUT}/fig2_class_contrasts_words.csv", index=False); kp.assign(dataset="phrase").to_csv(f"{OUT}/fig2_class_contrasts_phrases.csv", index=False)
    pd.set_option("display.width", 220)
    for t, lab in [(tw, "WORDS"), (tp, "PHRASES")]:
        n_pos = int((t.diff_mean > 0).sum()); n_sig = int(((t.diff_mean > 0) & (t.paired_t_p < .05)).sum()); n_rev = int(((t.diff_mean < 0) & (t.paired_t_p < .05)).sum())
        print(f"\n{lab}: {len(t)} domains, N terms = {int(t.N_terms.sum()):,}, {int(t.n_models.max())} models; abstract third more male in {n_pos}/{len(t)} "
              f"(two-tailed binomial P = {stats.binomtest(n_pos, len(t), 0.5).pvalue:.2g}); median gap {t.diff_mean.median():+.3f} SD; "
              f"paired-t P<0.05 in {n_sig}; significant reversals {n_rev}")
        print(t[["domain", "N_terms", "n_models", "cls", "concrete_mean", "abstract_mean", "diff_mean", "diff_ci", "paired_t_p"]].round(3).to_string(index=False))
    for c, k, lab in [(cw, kw, "WORDS"), (cp, kp, "PHRASES")]:
        print(f"\n{lab} construal-class centroids (mean across models, 95% CI across models):")
        print(c[c.model == "mean_across_models"][["cls", "N", "n_models", "conc_raw", "z_conc", "ci_conc", "z_gender", "ci_gender"]].round(3).to_string(index=False))
        print(k.round(4).to_string(index=False))
