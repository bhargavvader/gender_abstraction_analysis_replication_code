"""si_05_lexical_covariates.py  (= 04_covariates.py, 5 Oct 2026)

SUPPLEMENTARY SECTION 5 / SUPPLEMENTARY TABLES 5-8
'Robustness of the text-based results to lexical covariates and to the composition of the vocabulary'
(Supplementary Tables S-text-cov-words, S-text-cov-phrases, S-text-cov-permodel, S-text-composition), recomputed on the
12-model registry and the rebuilt master (Wikipedia GloVe word scores included).

Per model: OLS of z(gender) on z(concreteness) + covariates, HC1 robust SEs; the coefficient on concreteness is a
standardized beta (= Pearson r with no controls). Summaries across models: unweighted mean, 95% t CI, range, count negative,
count negative & Bonferroni-significant (k = number of models in the column).

Covariates. Words: log(1+SUBTLEX-US); length = characters and syllables (CMU dict, orthographic fallback); polysemy =
log(number of WordNet senses) for words with an entry; age of acquisition (Kuperman et al. 2012); valence, arousal,
dominance (Warriner et al. 2013); part of speech = Brysbaert dominant POS, else POS of the first WordNet sense, as fixed
effects. Phrases: mean of constituent-word covariates (frequency, AoA, VAD) over constituents in the respective norms;
length = characters and number of words; polysemy = WordNet senses of the phrase + has-entry indicator; POS = WordNet entry
POS, else head word's POS, as fixed effects.

Inputs: the FULL master; Brysbaert norms (SUBTLEX, Dom_Pos, Percent_known, Conc.SD); Kuperman AoA (AoAKuperman.csv, columns
Word, AoA_Kup); Warriner VAD (Ratings_Warriner_et_al.csv, columns Word, V/A/D.Mean.Sum); the Fig. 2 exclusion flags
(results/figure_02/all_terms_domains_flags.csv, from figure_02_a_domains_flags_classes.py); NLTK wordnet + cmudict.
Outputs: results/si_05/cov_*.csv, composition_*.csv, lodo_*.csv, words/phrases_with_covariates.csv.
"""
import os, sys, re, numpy as np, pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn, cmudict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MASTER, BRYS, AOA, VAD, results
from registry import REGISTRY
FLAGS  = os.path.join(results("figure_02"), "all_terms_domains_flags.csv")
OUT = results("si_05")
CONC = "conc_brysbaert"; MODELS = REGISTRY; GCOLS = [c for _, c in MODELS]; K = len(MODELS)

# ------------------------------------------------------------------ data ----------------------------------------
m = pd.read_csv(MASTER, low_memory=False, keep_default_na=False, na_values=["", "NA", "NaN"])
for c in [CONC, "SUBTLEX"] + GCOLS: m[c] = pd.to_numeric(m[c], errors="coerce")
m["key"] = m.term.astype(str).str.strip().str.lower()
fl = pd.read_csv(FLAGS, low_memory=False); fl["key"] = fl.term.astype(str).str.strip().str.lower()
fl = fl.drop_duplicates(["key", "dataset"])
m = m.merge(fl[["key", "dataset", "Dom_Pos", "wn_lexname", "wn_lex_method", "flag_noncontent", "flag_lexical_gender", "flag_person"]], on=["key", "dataset"], how="left")
b = pd.read_csv(BRYS); b["key"] = b.Word.astype(str).str.strip().str.lower(); b = b.drop_duplicates("key").set_index("key")
aoa = pd.read_csv(AOA, na_values=["none"]); aoa["key"] = aoa.Word.astype(str).str.lower(); aoa = aoa.dropna(subset=["AoA_Kup"]).drop_duplicates("key").set_index("key")["AoA_Kup"]
vad = pd.read_csv(VAD, index_col=0); vad["key"] = vad.Word.astype(str).str.lower(); vad = vad.drop_duplicates("key").set_index("key")[["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]]
cmu = cmudict.dict()
def syllables(w):
    if w in cmu: return max(1, sum(ch[-1].isdigit() for ch in cmu[w][0]))
    return max(1, len(re.findall(r"[aeiouy]+", w)))
WNPOS = {"n": "Noun", "v": "Verb", "a": "Adjective", "s": "Adjective", "r": "Adverb"}
def wn_first_pos(k):
    s = wn.synsets(k.replace(" ", "_")); return WNPOS.get(s[0].pos()) if s else np.nan

# ---- words
w = m[(m.dataset == "word") & m[CONC].notna()].copy()
w["logfreq"] = np.log1p(w.SUBTLEX)
w["nchar"] = w.key.str.len(); w["nsyll"] = w.key.map(lambda k: sum(syllables(t) for t in k.split()))
w["nsenses"] = w.key.map(lambda k: len(wn.synsets(k.replace(" ", "_"))))
w["logsenses"] = np.where(w.nsenses > 0, np.log(w.nsenses.clip(lower=1)), np.nan)
w["aoa"] = w.key.map(aoa); w = w.join(w.key.map(lambda k: vad.loc[k] if k in vad.index else pd.Series([np.nan]*3, index=vad.columns)).apply(pd.Series))
w = w.rename(columns={"V.Mean.Sum": "val", "A.Mean.Sum": "aro", "D.Mean.Sum": "dom"})
dp = w.key.map(b["Dom_Pos"]); dp = dp.where(dp.notna() & (dp.astype(str) != "0"), np.nan)
w["pos"] = dp.fillna(w.key.map(wn_first_pos))
w["pct_known"] = w.key.map(b["Percent_known"]); w["conc_sd"] = w.key.map(b["Conc.SD"])

# ---- phrases
p = m[(m.dataset == "phrase") & m[CONC].notna()].copy()
wlf = w.set_index("key")["logfreq"]; waoa = aoa; wvad = vad
def mean_over_constituents(k, series):
    vals = [series[t] for t in k.split() if t in series.index]
    return np.mean(vals) if vals else np.nan
p["logfreq"] = p.key.map(lambda k: mean_over_constituents(k, wlf))
p["aoa"] = p.key.map(lambda k: mean_over_constituents(k, waoa))
for col, nm in [("V.Mean.Sum", "val"), ("A.Mean.Sum", "aro"), ("D.Mean.Sum", "dom")]:
    s = wvad[col]; p[nm] = p.key.map(lambda k, s=s: mean_over_constituents(k, s))
p["nchar"] = p.key.str.len(); p["nwords"] = p.key.str.split().str.len()
p["nsenses"] = p.key.map(lambda k: len(wn.synsets(k.replace(" ", "_")))); p["has_entry"] = (p.nsenses > 0).astype(float)
p["logsenses"] = np.log1p(p.nsenses)
def phrase_pos(k):
    s = wn.synsets(k.replace(" ", "_"))
    if s: return WNPOS.get(s[0].pos())
    head = k.split()[-1] if k.split() else ""
    d = b["Dom_Pos"].get(head, np.nan)
    if isinstance(d, str) and d != "0": return d
    return wn_first_pos(head) if head else np.nan
p["pos"] = p.key.map(phrase_pos)
print(f"words: n={len(w):,}; WordNet entry {int((w.nsenses>0).sum()):,}; AoA {int(w.aoa.notna().sum()):,}; VAD {int(w.val.notna().sum()):,}; POS {int(w.pos.notna().sum()):,}")
print(f"phrases: n={len(p):,}; WordNet entry {int(p.has_entry.sum()):,}; freq {int(p.logfreq.notna().sum()):,}; AoA {int(p.aoa.notna().sum()):,}; VAD {int(p.val.notna().sum()):,}; POS {int(p.pos.notna().sum()):,}")
w.to_csv(f"{OUT}/words_with_covariates.csv", index=False); p.to_csv(f"{OUT}/phrases_with_covariates.csv", index=False)

# ------------------------------------------------------------------ estimation ----------------------------------
def z(x): return (x - x.mean()) / x.std(ddof=0)
def ols_hc1(y, X):
    X = np.column_stack([np.ones(len(y)), X]); n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X); beta = XtX_inv @ X.T @ y; e = y - X @ beta
    meat = (X * e[:, None]**2).T @ X; cov = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov)); t = beta / se; pv = 2 * stats.t.sf(np.abs(t), n - k)
    return beta[1], se[1], pv[1]            # coefficient on the first regressor (z-concreteness)
def fit(d, gcol, covs=(), pos_fe=False, sample=None, spearman=False):
    cols = [gcol, CONC] + list(covs) + (["pos"] if pos_fe else [])
    x = d[cols].dropna()
    if sample is not None: x = x[sample.reindex(x.index).fillna(False).astype(bool)]
    if len(x) < 100: return np.nan, np.nan, 0
    if spearman:
        r, pv = stats.spearmanr(x[gcol], x[CONC]); return r, pv, len(x)
    y = z(x[gcol]).values; X = [z(x[CONC]).values] + [z(x[c]).values if x[c].std() > 0 else x[c].values for c in covs]
    X = np.column_stack(X) if X else np.empty((len(x), 0))
    if pos_fe: X = np.column_stack([X, pd.get_dummies(x["pos"], drop_first=True).astype(float).values])
    beta, se, pv = ols_hc1(y, X); return beta, pv, len(x)
def summarize(rows):
    r = pd.DataFrame(rows); bb = r.beta.dropna(); k = len(bb); tcrit = stats.t.ppf(0.975, k - 1)
    tt = stats.ttest_1samp(bb, 0)
    return dict(mean=bb.mean(), ci_lo=bb.mean() - tcrit * bb.std(ddof=1) / np.sqrt(k), ci_hi=bb.mean() + tcrit * bb.std(ddof=1) / np.sqrt(k),
                min=bb.min(), max=bb.max(), k=k, k_neg=int((bb < 0).sum()), k_sig_neg=int(((r.beta < 0) & (r.p * k < 0.05)).sum()),
                t=tt.statistic, df=k - 1, p=tt.pvalue, n_min=int(r.n[r.n > 0].min()), n_max=int(r.n.max()))
def run_spec(d, label, covs=(), pos_fe=False, sample_fn=None, spearman=False, block=None):
    rows = []
    for name, gcol in MODELS:
        samp = sample_fn(d) if sample_fn is not None else None
        beta, pv, n = fit(d, gcol, covs, pos_fe, samp, spearman)
        rows.append(dict(model=name, beta=beta, p=pv, n=n))
    s = summarize(rows); s.update(spec=label, block=block or ""); return s, pd.DataFrame(rows).assign(spec=label)

specs_w = [
 ("A", "No controls (as in Fig.~1a; Supplementary Table~\\ref{tab:S-fig1-human})", dict()),
 ("A", "+ word frequency", dict(covs=["logfreq"])),
 ("A", "+ length (characters and syllables)", dict(covs=["nchar", "nsyll"])),
 ("A", "+ polysemy (number of WordNet senses)", dict(covs=["logsenses"])),
 ("A", "\\quad no controls, same words", dict(sample_fn=lambda d: d.logsenses.notna())),
 ("A", "+ age of acquisition\\cite{kuperman2012}", dict(covs=["aoa"])),
 ("A", "\\quad no controls, same words", dict(sample_fn=lambda d: d.aoa.notna())),
 ("A", "+ valence, arousal and dominance\\cite{warriner2013}", dict(covs=["val", "aro", "dom"])),
 ("A", "\\quad no controls, same words", dict(sample_fn=lambda d: d.val.notna())),
 ("A", "+ part of speech (fixed effects)", dict(pos_fe=True)),
 ("A", "+ frequency, length, polysemy and part of speech", dict(covs=["logfreq", "nchar", "nsyll", "logsenses"], pos_fe=True)),
 ("A", "+ all of the above (frequency, length, polysemy, age of acquisition, valence, arousal, dominance, part of speech)",
       dict(covs=["logfreq", "nchar", "nsyll", "logsenses", "aoa", "val", "aro", "dom"], pos_fe=True)),
 ("A", "\\quad no controls, same words", dict(sample_fn=lambda d: d.logsenses.notna() & d.aoa.notna() & d.val.notna() & d.pos.notna())),
 ("B", "Known by $\\geq$95\\% of raters", dict(sample_fn=lambda d: d.pct_known >= 0.95)),
 ("B", "Rater s.d.\\ of concreteness $\\leq$ median", dict(sample_fn=lambda d: d.conc_sd <= d.conc_sd.median())),
 ("B", "Both restrictions", dict(sample_fn=lambda d: (d.pct_known >= 0.95) & (d.conc_sd <= d.conc_sd.median()))),
]
w["fq"] = pd.qcut(w.logfreq.rank(method="first"), 5, labels=False)
for q in range(5):
    lab = "Quintile 1 (least frequent)" if q == 0 else "Quintile 5 (most frequent)" if q == 4 else f"Quintile {q+1}"
    specs_w.append(("C", lab, dict(sample_fn=(lambda d, q=q: d.fq == q))))
specs_p = [
 ("", "No controls (as in Fig.~1a; Supplementary Table~\\ref{tab:S-fig1-human})", dict()),
 ("", "+ word frequency (mean of constituent words)", dict(covs=["logfreq"])),
 ("", "+ length (characters and number of words)", dict(covs=["nchar", "nwords"])),
 ("", "+ polysemy (WordNet senses of the phrase; indicator for a WordNet entry)", dict(covs=["logsenses", "has_entry"])),
 ("", "+ age of acquisition (mean of constituent words)\\cite{kuperman2012}", dict(covs=["aoa"])),
 ("", "\\quad no controls, same phrases", dict(sample_fn=lambda d: d.aoa.notna())),
 ("", "+ valence, arousal and dominance (mean of constituent words)\\cite{warriner2013}", dict(covs=["val", "aro", "dom"])),
 ("", "\\quad no controls, same phrases", dict(sample_fn=lambda d: d.val.notna())),
 ("", "+ part of speech (fixed effects)", dict(pos_fe=True)),
 ("", "+ all of the above", dict(covs=["logfreq", "nchar", "nwords", "logsenses", "has_entry", "aoa", "val", "aro", "dom"], pos_fe=True)),
 ("", "\\quad no controls, same phrases", dict(sample_fn=lambda d: d.logfreq.notna() & d.aoa.notna() & d.val.notna() & d.pos.notna())),
]
sum_w, per_w = [], []
for blk, lab, kw in specs_w:
    s, r = run_spec(w, lab, block=blk, **kw); sum_w.append(s); per_w.append(r)
sum_p, per_p = [], []
for blk, lab, kw in specs_p:
    s, r = run_spec(p, lab, block=blk, **kw); sum_p.append(s); per_p.append(r)
sum_w, sum_p = pd.DataFrame(sum_w), pd.DataFrame(sum_p); per_w, per_p = pd.concat(per_w), pd.concat(per_p)

# ------------------------------------------------------------------ composition (Table S-text-composition) ---------
def extreme_mask(d, gcol, frac):
    zz = z(d[gcol]).abs(); cut = zz.quantile(1 - frac); return zz <= cut
def comp_specs(d, is_word):
    excl = ~(d.flag_person.fillna(False).astype(bool) | d.flag_lexical_gender.fillna(False).astype(bool) | (d.flag_noncontent.fillna(False).astype(bool) if is_word else False))
    ctrl = dict(covs=["logfreq", "nchar", "nsyll" if is_word else "nwords"], pos_fe=True)
    rows = []
    def spec(label, **kw):
        s, r = run_spec(d, label, **kw); rows.append(s)
    spec("All terms (as in Fig.~1a)")
    spec("Excluding person nouns, lexically gendered terms, function words and names", sample_fn=lambda d: excl)
    spec("As above, controlling for frequency, length and part of speech", sample_fn=lambda d: excl, **ctrl)
    # model-specific extreme exclusions
    for frac, lab in [(0.05, "Excluding the 5\\% most strongly gendered terms in each model"), (0.10, "Excluding the 10\\% most strongly gendered terms in each model")]:
        rr = [dict(model=n, **dict(zip(["beta", "p", "n"], fit(d, g, sample=extreme_mask(d, g, frac))))) for n, g in MODELS]
        s = summarize(rr); s.update(spec=lab); rows.append(s)
    rr = [dict(model=n, **dict(zip(["beta", "p", "n"], fit(d, g, sample=excl & extreme_mask(d, g, 0.05))))) for n, g in MODELS]
    s = summarize(rr); s.update(spec="Excluding person, gendered and function words and the 5\\% most gendered"); rows.append(s)
    spec("Spearman rank correlation, all terms", spearman=True)
    # leave-one-domain-out: least negative estimate per model (domains with >= 100 terms scored by the model)
    rr, worst = [], []
    for n, g in MODELS:
        x = d[[g, CONC, "wn_lexname"]].dropna(subset=[g, CONC]); doms = x.wn_lexname.value_counts(); doms = doms[doms >= 100].index
        ests = {}
        for dm in doms:
            xx = x[x.wn_lexname != dm]; ests[dm] = np.corrcoef(xx[g], xx[CONC])[0, 1]
        if not ests: rr.append(dict(model=n, beta=np.nan, p=np.nan, n=0)); continue
        dm_star = max(ests, key=ests.get); xx = x[x.wn_lexname != dm_star]
        r_, pv = stats.pearsonr(xx[g], xx[CONC]); rr.append(dict(model=n, beta=r_, p=pv, n=len(xx))); worst.append(dict(model=n, domain=dm_star, r=r_, n_domains=len(doms)))
    s = summarize(rr); s.update(spec="Leave-one-domain-out: least negative estimate for each model"); rows.append(s)
    return pd.DataFrame(rows), pd.DataFrame(worst)
comp_w, lodo_w = comp_specs(w, True); comp_p, lodo_p = comp_specs(p, False)

# ------------------------------------------------------------------ save ------------------------------------------
sum_w.to_csv(f"{OUT}/cov_summary_words.csv", index=False); sum_p.to_csv(f"{OUT}/cov_summary_phrases.csv", index=False)
per_w.to_csv(f"{OUT}/cov_per_model_words.csv", index=False); per_p.to_csv(f"{OUT}/cov_per_model_phrases.csv", index=False)
comp_w.to_csv(f"{OUT}/composition_words.csv", index=False); comp_p.to_csv(f"{OUT}/composition_phrases.csv", index=False)
lodo_w.to_csv(f"{OUT}/lodo_words.csv", index=False); lodo_p.to_csv(f"{OUT}/lodo_phrases.csv", index=False)
# descriptive correlations of concreteness with the covariates (words)
desc = {c: np.corrcoef(*w[[CONC, c]].dropna().values.T)[0, 1] for c in ["logfreq", "nchar", "nsyll", "aoa", "val", "aro"]}
pd.Series(desc).to_csv(f"{OUT}/conc_covariate_correlations_words.csv")
pd.set_option("display.width", 250)
print("\nconc-covariate r (words):", {k: round(v, 3) for k, v in desc.items()})
print("\nWORDS"); print(sum_w[["block", "spec", "mean", "ci_lo", "ci_hi", "min", "max", "k_neg", "k_sig_neg", "k", "t", "p", "n_min", "n_max"]].round(3).to_string(index=False))
print("\nPHRASES"); print(sum_p[["spec", "mean", "ci_lo", "ci_hi", "min", "max", "k_neg", "k_sig_neg", "k", "t", "p", "n_min", "n_max"]].round(3).to_string(index=False))
print("\nCOMPOSITION words"); print(comp_w[["spec", "mean", "ci_lo", "ci_hi", "k_neg", "k_sig_neg", "k"]].round(3).to_string(index=False))
print("\nCOMPOSITION phrases"); print(comp_p[["spec", "mean", "ci_lo", "ci_hi", "k_neg", "k_sig_neg", "k"]].round(3).to_string(index=False))
print("\nLODO words:", lodo_w.domain.value_counts().to_dict(), "| phrases:", lodo_p.domain.value_counts().to_dict(), "| n domains:", lodo_w.n_domains.unique(), lodo_p.n_domains.unique())
