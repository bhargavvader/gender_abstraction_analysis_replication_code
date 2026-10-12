"""si_07_domain_assignment_sensitivity.py  (= 05_domain_sensitivity.py, 5 Oct 2026)

SUPPLEMENTARY SECTION 7 / SUPPLEMENTARY TABLE 10
'Sensitivity of the within-domain results to the WordNet domain assignment', recomputed on the
12-model registry. Fixed vocabulary: single-word entries of the concreteness norms whose Brysbaert dominant POS is noun, verb,
adjective or adverb and that have >= 1 WordNet sense in that POS (adj.ppl folded into adj.all; noun.person domain excluded).
Schemes: primary sense, all senses, domain-unambiguous words only, random sense (200 replicates, seed 2026). Within each
scheme: domains with >= 100 assigned words; terciles by concreteness; per-model mean z (z-scored over the fixed vocabulary the
model covers) of abstract and concrete thirds; equal-weight mean gap across models; one-sample t-test across models; binomial
sign test across domains. Also recomputes the Supplementary Fig. 1 panel correlations under the primary-sense and all-senses rules.

Inputs: the FULL master, the Brysbaert norms (Dom_Pos); NLTK wordnet. Outputs: results/si_07/domain_sensitivity_*.csv,
example_panels_primary_vs_allsenses.csv."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MASTER, BRYS, results
from registry import REGISTRY
OUT = results("si_07")
CONC = "conc_brysbaert"; MIN_DOM = 100; SEED = 2026; NREP = 200
POS_MAP = {"Noun": ["n"], "Verb": ["v"], "Adjective": ["a", "s"], "Adverb": ["r"]}

m = pd.read_csv(MASTER, low_memory=False, keep_default_na=False, na_values=["", "NA", "NaN"])
GCOLS = [c for _, c in REGISTRY]
for c in [CONC] + GCOLS: m[c] = pd.to_numeric(m[c], errors="coerce")
m["key"] = m.term.astype(str).str.strip().str.lower()
b = pd.read_csv(BRYS); b["key"] = b.Word.astype(str).str.strip().str.lower(); b = b.drop_duplicates("key").set_index("key")
w = m[(m.dataset == "word") & m[CONC].notna() & ~m.key.str.contains(" ")].copy()
w["Dom_Pos"] = w.key.map(b["Dom_Pos"]); w = w[w.Dom_Pos.isin(POS_MAP)]
def senses(row):
    syns = [s for s in wn.synsets(row.key) if s.pos() in POS_MAP[row.Dom_Pos]]
    return [("adj.all" if s.lexname() == "adj.ppl" else s.lexname()) for s in syns]
w["lexnames"] = w.apply(senses, axis=1); w = w[w.lexnames.map(len) > 0].reset_index(drop=True)
w["primary"] = w.lexnames.map(lambda L: L[0]); w["n_distinct"] = w.lexnames.map(lambda L: len(set(L)))
print(f"fixed vocabulary: {len(w):,} words; domain-unambiguous: {int((w.n_distinct==1).sum()):,}")
Z = {n: (w[c] - w[c].mean()) / w[c].std(ddof=0) for n, c in REGISTRY}          # each model z-scored over the vocabulary it covers (NaN preserved)

def analyse(assign):
    """assign: list (len = len(w)) of lists of domains for each word (one or several)."""
    rows = [(i, d) for i, L in enumerate(assign) for d in L]
    A = pd.DataFrame(rows, columns=["i", "domain"]); A = A[A.domain != "noun.person"]
    counts = A.domain.value_counts(); keep = counts[counts >= MIN_DOM].index; A = A[A.domain.isin(keep)]
    out = []
    for dom, g in A.groupby("domain"):
        idx = g.i.values; conc = w[CONC].values[idx]
        lo, hi = np.quantile(conc, [1/3, 2/3]); ab = idx[conc <= lo]; co = idx[conc >= hi]
        gaps = []
        for n, _ in REGISTRY:
            za = Z[n].values[ab]; zc = Z[n].values[co]; za = za[~np.isnan(za)]; zc = zc[~np.isnan(zc)]
            if len(za) < 20 or len(zc) < 20: continue
            gaps.append(za.mean() - zc.mean())
        gaps = np.array(gaps); tt = stats.ttest_1samp(gaps, 0)
        out.append(dict(domain=dom, n_words=len(idx), n_models=len(gaps), gap=gaps.mean(), p=tt.pvalue))
    t = pd.DataFrame(out)
    n_pos = int((t.gap > 0).sum()); summ = dict(words_assigned=len(A), domains=len(t), n_pos=n_pos,
        sign_p=stats.binomtest(n_pos, len(t), 0.5).pvalue, median_gap=t.gap.median(),
        sig_pos=int(((t.gap > 0) & (t.p < .05)).sum()), sig_neg=int(((t.gap < 0) & (t.p < .05)).sum()))
    return summ, t

results_ = {}
results_["primary"], t_primary = analyse([[d] for d in w.primary])
results_["all_senses"], t_all = analyse([sorted(set(L)) for L in w.lexnames])
unamb = w.n_distinct == 1
results_["unambiguous"], t_unamb = analyse([[L[0]] if u else [] for L, u in zip(w.lexnames, unamb)])
rng = np.random.default_rng(SEED); reps = []
for r in range(NREP):
    s, _ = analyse([[L[rng.integers(len(L))]] for L in w.lexnames]); s["rep"] = r; reps.append(s)
reps = pd.DataFrame(reps)
summary = pd.DataFrame([dict(scheme=k, **v) for k, v in results_.items()])
summary.to_csv(f"{OUT}/domain_sensitivity_summary.csv", index=False); reps.to_csv(f"{OUT}/domain_sensitivity_random_reps.csv", index=False)
pd.concat([t_primary.assign(scheme="primary"), t_all.assign(scheme="all_senses"), t_unamb.assign(scheme="unambiguous")]).to_csv(f"{OUT}/domain_sensitivity_per_domain.csv", index=False)
pd.set_option("display.width", 200); print(summary.round(4).to_string(index=False))
print(f"random ({NREP} reps): domains {reps.domains.min()}-{reps.domains.max()}; n_pos {reps.n_pos.min()}-{reps.n_pos.max()} "
      f"(share {reps.n_pos.div(reps.domains).min():.0%}-{reps.n_pos.div(reps.domains).max():.0%}, median {reps.n_pos.div(reps.domains).median():.0%}); "
      f"max sign P {reps.sign_p.max():.2g}; median gap {reps.median_gap.min():.2f}-{reps.median_gap.max():.2f}; sig+ {reps.sig_pos.min()}-{reps.sig_pos.max()}; "
      f"sig- {reps.sig_neg.min()}-{reps.sig_neg.max()}; reps with no sig reversal {(reps.sig_neg==0).sum()}; max reversals {reps.sig_neg.max()}")

# example panels: primary-sense vs all-senses within-domain correlations for the three SI example models
EX_MODELS = ["Qwen3", "GPT4", "Nomic v2 MoE"]; EX_DOMAINS = ["verb.communication", "adj.all", "noun.artifact"]
ex = []
for mod in EX_MODELS:
    col = dict(REGISTRY)[mod]
    for dom in EX_DOMAINS:
        for scheme, mask in [("primary", w.primary == dom), ("all_senses", w.lexnames.map(lambda L: dom in L))]:
            x = w.loc[mask, [col, CONC]].dropna(); r_, pv = stats.pearsonr(x[col], x[CONC])
            ex.append(dict(model=mod, domain=dom, scheme=scheme, n=len(x), r=r_, p=pv))
ex = pd.DataFrame(ex); ex.to_csv(f"{OUT}/example_panels_primary_vs_allsenses.csv", index=False)
print(ex.pivot(index=["domain", "model"], columns="scheme", values=["r", "n", "p"]).round(3).to_string())
