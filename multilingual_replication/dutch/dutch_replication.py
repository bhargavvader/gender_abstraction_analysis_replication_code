#!/usr/bin/env python3
"""
Dutch replication of the male abstraction bias (Fig. 1a) -- script form of dutch_replication.ipynb.
Run from the folder that holds nl-brysbaert-2014.tsv:

    pip install -r requirements.txt
    pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_lg-3.8.0/nl_core_news_lg-3.8.0-py3-none-any.whl
    python dutch_replication.py

Writes results_nl/ (stats, deciles, seed-pair sensitivity, per-word scores, seeds, held-out validation and
figS_dutch_replication.{png,pdf}). The notebook and this script contain the same code and produce identical files.
"""
import matplotlib
matplotlib.use("Agg")

def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)

# ---------------------------------------------------------------- configuration
from pathlib import Path
import re

BASE  = Path(__file__).resolve().parent             # the folder holding this script and the norms file
NORMS = BASE / "nl-brysbaert-2014.tsv"
OUT   = BASE / "results_nl"; OUT.mkdir(exist_ok=True)
assert NORMS.exists(), f"missing file: {NORMS}"

LANGNAME, SCALE, SPACY_MODEL, WORDFREQ_LANG = "Dutch", "1–5", "nl_core_news_lg", "nl"
SEEDS = {
    "primary":  [("man", "vrouw"), ("jongen", "meisje"), ("mannelijk", "vrouwelijk"), ("vader", "moeder"), ("broer", "zus")],
    "extended": [("man", "vrouw"), ("jongen", "meisje"), ("mannelijk", "vrouwelijk"), ("vader", "moeder"), ("broer", "zus"),
                 ("zoon", "dochter"), ("oom", "tante"), ("koning", "koningin"), ("heer", "dame"), ("opa", "oma")],
    "pronoun":  [("man", "vrouw"), ("hij", "zij"), ("hem", "haar"), ("mannelijk", "vrouwelijk"), ("jongen", "meisje")],
}
# gendered word pairs not used as primary seeds: a valid axis orders the male above the female member
VALIDATION = [("koning", "koningin"), ("oom", "tante"), ("zoon", "dochter"), ("acteur", "actrice"), ("heer", "dame"), ("opa", "oma"),
              ("neef", "nicht"), ("prins", "prinses"), ("bruidegom", "bruid"), ("meneer", "mevrouw"), ("kerel", "meid"), ("monnik", "non"),
              ("held", "heldin"), ("vriend", "vriendin"), ("echtgenoot", "echtgenote"), ("leraar", "lerares"), ("zanger", "zangeres"),
              ("weduwnaar", "weduwe"), ("stiefvader", "stiefmoeder"), ("grootvader", "grootmoeder"), ("papa", "mama"), ("jongens", "meisjes"),
              ("mannen", "vrouwen"), ("vaders", "moeders"), ("broers", "zussen"), ("hij", "zij"), ("hem", "haar")]
# crude 'lexically gendered' filter (gendered morphemes, kinship and role stems), used in one exclusion check
GENDERED_RE = re.compile(r"(?:man$|mannen|vrouw|jongen|meisje|vader|moeder|broer|zus$|zuster|heer$|dame|koning|prins|mannelijk|vrouwelijk|-in$|ster$|es$|vrouwelijk|mama|papa|oma|opa|tante|oom|neef|nicht|dochter|zoon|echtgeno|weduw|bruid|bruidegom|non$|monnik|priester|meid|knaap|kerel|vent$|wijf|gast$|jongen|meisje|juffrouw|mevrouw|mijnheer|meneer)")
print("norms:", NORMS.name, "| outputs:", OUT.name)

# ---------------------------------------------------------------- imports and helpers
import json
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 80)

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

_warned = set()
def gender_axis(lookup, pairs):
    # Eq. S1: mean over seed pairs of unit(male) - unit(female), re-normalised
    diffs, used = [], []
    for m, f in pairs:
        vm, vf = lookup(m), lookup(f)
        if vm is None or vf is None:
            if (m, f) not in _warned: print(f"   seed pair missing in the vectors: ({m}, {f})"); _warned.add((m, f))
            continue
        diffs.append(unit(vm) - unit(vf)); used.append((m, f))
    if not diffs: raise RuntimeError("no seed pairs available in the vectors")
    return unit(np.mean(diffs, axis=0)), used

def score_terms(lookup, axis, words):
    g = np.full(len(words), np.nan)
    for i, w in enumerate(words):
        v = lookup(w)
        if v is not None: g[i] = float(unit(v) @ axis)
    return g

def fisher_ci(r, n):
    if n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3); q = stats.norm.ppf(0.975)
    return tuple(np.tanh([z - q * se, z + q * se]))

def zs(x):
    x = np.asarray(x, float); return (x - np.nanmean(x)) / np.nanstd(x)

def describe(df, label):
    d = df.dropna(subset=["gender", "conc"])
    r, p = stats.pearsonr(d.gender, d.conc); lo, hi = fisher_ci(r, len(d)); rho, _ = stats.spearmanr(d.gender, d.conc)
    return dict(specification=label, r=r, ci_lo=lo, ci_hi=hi, p=p, rho=rho, n=len(d))

def ols_beta(df, controls, label):
    # standardised coefficient on concreteness (HC1 SEs) in a regression of z(gender) on z(concreteness) + controls
    cols = ["conc"] + controls
    d = df.dropna(subset=["gender"] + cols)
    X = sm.add_constant(np.column_stack([zs(d[c]) for c in cols]))
    res = sm.OLS(zs(d.gender), X).fit(cov_type="HC1"); ci = res.conf_int()[1]
    return dict(specification=label, r=res.params[1], ci_lo=ci[0], ci_hi=ci[1], p=res.pvalues[1], rho=np.nan, n=len(d))

def deciles(df):
    d = df.dropna(subset=["gender", "conc"]).copy()
    d["decile"] = pd.qcut(d.gender.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    return d.groupby("decile").conc.agg(["mean", "count"]).reset_index()

def validation(lookup, axis, used):
    rows = []
    for m, f in VALIDATION:
        if (m, f) in used: continue
        vm, vf = lookup(m), lookup(f)
        if vm is None or vf is None: continue
        sm_, sf_ = float(unit(vm) @ axis), float(unit(vf) @ axis)
        rows.append(dict(male=m, female=f, score_male=sm_, score_female=sf_, male_above_female=sm_ > sf_))
    return pd.DataFrame(rows)

from wordfreq import zipf_frequency
raw = pd.read_csv(NORMS, sep="\t")
norms = raw.rename(columns={"concreteness": "conc", "age of acquisition": "aoa"})
norms["word"] = norms["word"].astype(str).str.strip()
norms["conc"] = pd.to_numeric(norms["conc"], errors="coerce")
norms = norms.dropna(subset=["conc"]).drop_duplicates("word").reset_index(drop=True)   # keep the first entry of a repeated form
norms["length"] = norms["word"].str.len()
norms["logfreq"] = [zipf_frequency(w, WORDFREQ_LANG) for w in norms["word"]]
norms.loc[norms["logfreq"] <= 0, "logfreq"] = np.nan                                   # no wordfreq entry -> missing
print(f"{len(raw):,} rows; {len(norms):,} distinct rated words; concreteness mean {norms.conc.mean():.2f} (sd {norms.conc.std():.2f}) on a {SCALE} scale; "
      f"{norms.logfreq.notna().sum():,} with a frequency, {norms.aoa.notna().sum():,} with an AoA rating")
print(norms.sample(8, random_state=1).to_string())

import spacy
nlp = spacy.load(SPACY_MODEL, exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "morphologizer", "senter"])
print(SPACY_MODEL, "v" + nlp.meta["version"], "| vectors", nlp.vocab.vectors.shape, "| sources:", [s.get("name") for s in nlp.meta.get("sources", [])])

def lookup(w):
    for cand in (w, w.lower()):
        lex = nlp.vocab[cand]
        if lex.has_vector and lex.vector_norm > 0: return np.asarray(lex.vector, dtype=np.float32)
    return None

covered = norms.word.map(lambda w: lookup(w) is not None)
print(f"{covered.sum():,} of {len(norms):,} rated words ({covered.mean():.1%}) have a vector")
missing = sorted({p for s in SEEDS.values() for p in s if lookup(p[0]) is None or lookup(p[1]) is None})
print("seed pairs missing from the vectors:", missing or "none")

rows, dec_tables, seeds_used, scores, VAL = [], [], {}, norms[["word", "conc"]].copy(), []
for pset, pairs in SEEDS.items():
    axis, used = gender_axis(lookup, pairs); seeds_used[pset] = used
    g = score_terms(lookup, axis, norms.word.tolist())
    df = norms.copy(); df["gender"] = g; scores[f"gender::{pset}"] = g
    V = validation(lookup, axis, used); V["seeds"] = pset; VAL.append(V)
    print(f"seeds={pset}: {len(used)}/{len(pairs)} pairs; {int(np.isfinite(g).sum()):,}/{len(df):,} words scored; "
          f"held-out gendered pairs ordered male > female: {int(V.male_above_female.sum())}/{len(V)}")
    R = [describe(df, "all terms (no controls)")]
    seedset = {w for p in used for w in p}
    R.append(describe(df[~(df.word.str.contains(GENDERED_RE) | df.word.isin(seedset))], "excl. lexically gendered terms + seeds"))
    for q in (0.05, 0.10):
        thr = df.gender.abs().quantile(1 - q); R.append(describe(df[df.gender.abs() < thr], f"excl. {int(q*100)}% most gendered terms"))
    R.append(ols_beta(df, [], "OLS: no controls"))
    R.append(ols_beta(df, ["logfreq"], "OLS: + log frequency"))
    R.append(ols_beta(df, ["length"], "OLS: + length"))
    R.append(ols_beta(df, ["logfreq", "length"], "OLS: + frequency + length"))
    R.append(ols_beta(df, ["logfreq", "length", "aoa"], "OLS: + frequency + length + AoA"))
    dq = df.dropna(subset=["logfreq", "gender"]).copy()
    dq["fq"] = pd.qcut(dq.logfreq.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    for k in range(1, 6):
        R.append(describe(dq[dq.fq == k], f"frequency quintile {k}" + (" (least frequent)" if k == 1 else " (most frequent)" if k == 5 else "")))
    for r_ in R: r_.update(seeds=pset)
    rows += R
    dd = deciles(df); dd["seeds"] = pset; dec_tables.append(dd)

S = pd.DataFrame(rows)[["seeds", "specification", "r", "ci_lo", "ci_hi", "p", "rho", "n"]]
S.insert(0, "model", SPACY_MODEL); S.insert(0, "language", LANGNAME)
DEC = pd.concat(dec_tables, ignore_index=True); VAL = pd.concat(VAL, ignore_index=True)
print(S.drop(columns=["language", "model"]).round(4).to_string(index=False))

print("Mean concreteness by gender-score decile (1 = most feminine, 10 = most masculine):")
display(DEC.pivot_table(index="decile", columns="seeds", values="mean").round(3))

n_words = int(S[(S.seeds == "primary") & (S.specification == "all terms (no controls)")].n.iloc[0])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.4, 4.3))
for pset, label, col, mk in [("primary", "5 seed pairs (primary)", "#1f77b4", "o"), ("extended", "10 seed pairs (extended)", "#17becf", "s")]:
    d = DEC[DEC.seeds == pset].sort_values("decile"); ax.plot(d.decile, d["mean"], marker=mk, ms=4, lw=1.4, color=col, label=label)
ax.set_xticks(range(1, 11)); ax.set_xlim(0.5, 10.5)
ax.set_ylabel(f"Mean Concreteness Rating\n({LANGNAME} norms, {SCALE})")
ax.set_title(f"{LANGNAME}: Human Judgments of Concreteness", fontsize=10); ax.grid(alpha=0.25, lw=0.5)
ax.text(0.03, 0.03, f"N = {n_words:,} words", ha="left", va="bottom", transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.6", lw=0.6))
ax.legend(loc="upper right", fontsize=8, frameon=False, title="Gender axis (spaCy nl vectors)", title_fontsize=8)
ax.set_xlabel("Gender Score Decile", labelpad=34)
ax.annotate("", xy=(0.02, -0.11), xytext=(0.44, -0.11), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("", xy=(0.98, -0.11), xytext=(0.56, -0.11), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.23, -0.14, "More Feminine", ha="center", va="top", transform=ax.transAxes, fontsize=8)
ax.text(0.77, -0.14, "More Masculine", ha="center", va="top", transform=ax.transAxes, fontsize=8)
for ext in ("png", "pdf"): fig.savefig(OUT / f"figS_dutch_replication.{ext}", dpi=300, bbox_inches="tight")
print("figure written to", OUT / "figS_dutch_replication.png")

prim, ext = SEEDS["primary"], SEEDS["extended"]
specs = {"primary set (5 pairs)": prim, "extended set (10 pairs)": ext}
for p in prim: specs[f"primary minus {p[0]}/{p[1]}"] = [q for q in prim if q != p]
for p in ext: specs[f"single pair {p[0]}/{p[1]}"] = [p]
specs["pronoun set"] = SEEDS["pronoun"]
for p in SEEDS["pronoun"]:
    if p not in ext: specs[f"single pair {p[0]}/{p[1]}"] = [p]
PS = []
for label, pairs in specs.items():
    axis, used = gender_axis(lookup, pairs)
    g = score_terms(lookup, axis, norms.word.tolist()); ok = np.isfinite(g)
    r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
    d = pd.DataFrame({"w": norms.word, "g": g}).dropna().sort_values("g")
    PS.append(dict(specification=label, n_pairs=len(used), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()),
                   most_female=", ".join(d.w.head(8)), most_male=", ".join(d.w.tail(8)[::-1])))
D = np.vstack([unit(unit(lookup(m)) - unit(lookup(f))) for m, f in ext])
U, s, Vt = np.linalg.svd(D, full_matrices=False); pc1 = Vt[0]; pc1 = pc1 if D.mean(0) @ pc1 > 0 else -pc1
g = score_terms(lookup, unit(pc1), norms.word.tolist()); ok = np.isfinite(g)
r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
d = pd.DataFrame({"w": norms.word, "g": g}).dropna().sort_values("g")
PS.append(dict(specification=f"PC1 of extended set (var. expl. {s[0]**2/(s**2).sum():.2f}; loadings {np.round(D@pc1,2).tolist()})",
               n_pairs=len(D), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()), most_female=", ".join(d.w.head(8)), most_male=", ".join(d.w.tail(8)[::-1])))
PS = pd.DataFrame(PS)
print(PS[["specification", "n_pairs", "r", "ci_lo", "ci_hi", "n", "most_female", "most_male"]].round(3).to_string(index=False))

S.to_csv(OUT / "stats_nl.csv", index=False)
DEC.to_csv(OUT / "deciles_nl.csv", index=False)
PS.to_csv(OUT / "pole_sensitivity_nl.csv", index=False)
scores.to_csv(OUT / "scores_nl.csv", index=False)
VAL.to_csv(OUT / "validation_nl.csv", index=False)
json.dump(seeds_used, open(OUT / "seeds_nl.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("written to", OUT.name)
for f in sorted(OUT.iterdir()): print("  ", f.name)
