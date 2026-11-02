#!/usr/bin/env python3
"""
Chinese (Mandarin) replication of the male abstraction bias (Fig. 1a) -- script form of
chinese_replication.ipynb. Run from the folder that holds zh-chan-tse-2024.csv and zh-song-li-2021.csv:

    pip install -r requirements.txt
    pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-3.8.0/zh_core_web_lg-3.8.0-py3-none-any.whl
    python chinese_replication.py

Writes results_zh/ (Chan & Tse 2024 norms: stats, deciles, seed-pair sensitivity, per-word scores, seeds,
figS_chinese_replication.{png,pdf}) and results_zh_song/ (Song & Li 2021 Mainland Mandarin norms).
The notebook and this script contain the same code and produce identical files.
"""
import matplotlib
matplotlib.use("Agg")

def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)

# ---------------------------------------------------------------- configuration
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent              # the folder holding this script and the two norm files
NORMS_CHAN = BASE / "zh-chan-tse-2024.csv"          # Chan & Tse (2024) — main analysis
NORMS_SONG = BASE / "zh-song-li-2021.csv"           # Song & Li (2021)  — Mainland Mandarin check
OUT_CHAN, OUT_SONG = BASE / "results_zh", BASE / "results_zh_song"
for p in (NORMS_CHAN, NORMS_SONG): assert p.exists(), f"missing file: {p}"
OUT_CHAN.mkdir(exist_ok=True); OUT_SONG.mkdir(exist_ok=True)

LANGNAME, SCALE, SPACY_MODEL = "Chinese", "1–7", "zh_core_web_lg"
SEEDS = {
    "primary":  [("男人", "女人"), ("男孩", "女孩"), ("男性", "女性"), ("父亲", "母亲"), ("哥哥", "姐姐")],
    "extended": [("男人", "女人"), ("男孩", "女孩"), ("男性", "女性"), ("父亲", "母亲"), ("哥哥", "姐姐"),
                 ("儿子", "女儿"), ("叔叔", "阿姨"), ("国王", "女王"), ("先生", "女士"), ("爷爷", "奶奶")],
    "pronoun":  [("男人", "女人"), ("他", "她"), ("男性", "女性"), ("男孩", "女孩"), ("父亲", "母亲")],
}
# gendered word pairs never used as seeds: a valid axis orders the male above the female member
VALIDATION = [("丈夫", "妻子"), ("弟弟", "妹妹"), ("爸爸", "妈妈"), ("新郎", "新娘"), ("王子", "公主"),
              ("少年", "少女"), ("男子", "女子"), ("公公", "婆婆"), ("外公", "外婆"), ("孙子", "孙女"),
              ("侄子", "侄女"), ("男生", "女生"), ("绅士", "淑女"), ("男友", "女友"), ("和尚", "尼姑"),
              ("皇帝", "皇后"), ("帅哥", "美女"), ("男神", "女神")]
# words containing a gender-marking character count as 'lexically gendered' for the exclusion check
GENDERED_RE = re.compile(r"[男女父母爸妈哥姐弟妹夫妻公婆郎娘姑叔舅姨翁妇嫂爷奶兄嬷媳婿姥婶他她]")
print("norms:", NORMS_CHAN.name, "+", NORMS_SONG.name, "| outputs:", OUT_CHAN.name, "+", OUT_SONG.name)

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

def validation_accuracy(lookup, axis, used):
    hits = []
    for m, f in VALIDATION:
        if (m, f) in used: continue
        vm, vf = lookup(m), lookup(f)
        if vm is None or vf is None: continue
        hits.append(float(unit(vm) @ axis) > float(unit(vf) @ axis))
    return (np.mean(hits) if hits else np.nan), len(hits)

def load_norms(path, word_col, conc_col, sd_col=None, freq_col=None, log_freq=False):
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    norms = pd.DataFrame({"word": raw[word_col].astype(str).str.strip(),
                          "conc": pd.to_numeric(raw[conc_col], errors="coerce"),
                          "conc_sd": pd.to_numeric(raw[sd_col], errors="coerce") if sd_col else np.nan,
                          "norm_freq": pd.to_numeric(raw[freq_col], errors="coerce") if freq_col else np.nan})
    if log_freq: norms["norm_freq"] = np.log1p(norms["norm_freq"])
    norms = norms[(norms.word != "") & norms.conc.notna()]
    norms = norms.groupby("word", as_index=False).agg(conc=("conc", "mean"), conc_sd=("conc_sd", "mean"), norm_freq=("norm_freq", "mean"))
    norms["length"] = norms.word.str.len()
    norms["logfreq"] = norms["norm_freq"]          # log corpus frequency distributed with the norms
    return norms

import spacy
nlp = spacy.load(SPACY_MODEL, exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "morphologizer", "senter"])
print(SPACY_MODEL, "v" + nlp.meta["version"], "| vectors", nlp.vocab.vectors.shape, "| sources:", [s.get("name") for s in nlp.meta.get("sources", [])])

def lookup(w):
    for cand in (w, w.lower()):
        lex = nlp.vocab[cand]
        if lex.has_vector and lex.vector_norm > 0: return np.asarray(lex.vector, dtype=np.float32)
    return None

missing = sorted({p for s in SEEDS.values() for p in s if lookup(p[0]) is None or lookup(p[1]) is None})
print("seed pairs missing from the vectors:", missing or "none")

def run_all(norms, tag, out):
    rows, dec, seeds_used, scores = [], [], {}, norms[["word", "conc"]].copy()
    for pset, pairs in SEEDS.items():
        axis, used = gender_axis(lookup, pairs); seeds_used[pset] = used
        g = score_terms(lookup, axis, norms.word.tolist())
        df = norms.copy(); df["gender"] = g; scores[f"gender::{pset}"] = g
        acc, nval = validation_accuracy(lookup, axis, used)
        print(f"  seeds={pset}: {len(used)}/{len(pairs)} pairs; {int(np.isfinite(g).sum()):,}/{len(df):,} words scored; "
              f"held-out gendered pairs ordered correctly: {acc:.2f} (n={nval})")
        R = [describe(df, "all terms (no controls)")]
        seedset = {w for p in used for w in p}
        R.append(describe(df[~(df.word.str.contains(GENDERED_RE) | df.word.isin(seedset))], "excl. lexically gendered terms + seeds"))
        for q in (0.05, 0.10):
            thr = df.gender.abs().quantile(1 - q); R.append(describe(df[df.gender.abs() < thr], f"excl. {int(q*100)}% most gendered terms"))
        if df.conc_sd.notna().sum() > 100:
            R.append(describe(df[df.conc_sd <= df.conc_sd.median()], "rater s.d. of concreteness <= median (reliably rated)"))
        if df.logfreq.notna().sum() > 100:
            R.append(ols_beta(df, ["logfreq"], "OLS: + log frequency"))
        R.append(ols_beta(df, [], "OLS: no controls"))
        for r_ in R: r_.update(seeds=pset, validation_accuracy=acc, validation_pairs=nval)
        rows += R
        dd = deciles(df); dd["seeds"] = pset; dec.append(dd)
    # seed-pair sensitivity
    prim, ext = SEEDS["primary"], SEEDS["extended"]
    specs = {"primary set (5 pairs)": prim, "extended set (10 pairs)": ext}
    for p in prim: specs[f"primary minus {p[0]}/{p[1]}"] = [q for q in prim if q != p]
    for p in ext: specs[f"single pair {p[0]}/{p[1]}"] = [p]
    for k, v in SEEDS.items():
        if k not in ("primary", "extended"):
            specs[f"{k} set"] = v
            for p in v:
                if p not in ext: specs[f"single pair {p[0]}/{p[1]}"] = [p]
    PS = []
    for label, pairs in specs.items():
        try: axis, used = gender_axis(lookup, pairs)
        except RuntimeError: continue
        g = score_terms(lookup, axis, norms.word.tolist()); ok = np.isfinite(g)
        r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
        d = pd.DataFrame({"w": norms.word, "g": g}).dropna().sort_values("g")
        PS.append(dict(specification=label, n_pairs=len(used), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()),
                       most_female=", ".join(d.w.head(8)), most_male=", ".join(d.w.tail(8)[::-1])))
    D = np.vstack([unit(unit(lookup(m)) - unit(lookup(f))) for m, f in ext if lookup(m) is not None and lookup(f) is not None])
    U, s, Vt = np.linalg.svd(D, full_matrices=False); pc1 = Vt[0]; pc1 = pc1 if D.mean(0) @ pc1 > 0 else -pc1
    g = score_terms(lookup, unit(pc1), norms.word.tolist()); ok = np.isfinite(g)
    r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
    d = pd.DataFrame({"w": norms.word, "g": g}).dropna().sort_values("g")
    PS.append(dict(specification=f"PC1 of extended set (var. expl. {s[0]**2/(s**2).sum():.2f}; loadings {np.round(D@pc1,2).tolist()})",
                   n_pairs=len(D), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()),
                   most_female=", ".join(d.w.head(8)), most_male=", ".join(d.w.tail(8)[::-1])))
    # write
    S = pd.DataFrame(rows)[["seeds", "specification", "r", "ci_lo", "ci_hi", "p", "rho", "n", "validation_accuracy", "validation_pairs"]]
    S.insert(0, "model", SPACY_MODEL); S.insert(0, "language", LANGNAME)
    DEC = pd.concat(dec); PS = pd.DataFrame(PS)
    S.to_csv(out / f"stats_{tag}.csv", index=False); DEC.to_csv(out / f"deciles_{tag}.csv", index=False)
    PS.to_csv(out / f"pole_sensitivity_{tag}.csv", index=False); scores.to_csv(out / f"scores_{tag}.csv", index=False)
    json.dump(seeds_used, open(out / f"seeds_{tag}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    return S, DEC, PS

norms = load_norms(NORMS_CHAN, word_col="word_chinese_simplified", conc_col="concrete_mean", sd_col="concrete_sd", freq_col="frequency_log_word")
print(f"{len(norms):,} distinct rated words; concreteness mean {norms.conc.mean():.2f} (sd {norms.conc.std():.2f}) on a {SCALE} scale")
S, DEC, PS = run_all(norms, "zh", OUT_CHAN)
print(S.drop(columns=["language", "model"]).round(4).to_string(index=False))

print("Mean concreteness by gender-score decile (1 = most feminine, 10 = most masculine):")
display(DEC.pivot_table(index="decile", columns="seeds", values="mean").round(3))
print(PS[["specification", "n_pairs", "r", "ci_lo", "ci_hi", "n", "most_female", "most_male"]].round(3).to_string(index=False))

n_words = int(S[(S.seeds == "primary") & (S.specification == "all terms (no controls)")].n.iloc[0])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.4, 4.3))
for pset, label, col, mk in [("primary", "5 seed pairs (primary)", "#1f77b4", "o"), ("extended", "10 seed pairs (extended)", "#17becf", "s")]:
    d = DEC[DEC.seeds == pset].sort_values("decile"); ax.plot(d.decile, d["mean"], marker=mk, ms=4, lw=1.4, color=col, label=label)
ax.set_xticks(range(1, 11)); ax.set_xlim(0.5, 10.5)
ax.set_ylabel(f"Mean Concreteness Rating\n({LANGNAME} norms, {SCALE})")
ax.set_title(f"{LANGNAME}: Human Judgments of Concreteness", fontsize=10); ax.grid(alpha=0.25, lw=0.5)
ax.text(0.03, 0.03, f"N = {n_words:,} words", ha="left", va="bottom", transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.6", lw=0.6))
ax.legend(loc="best", fontsize=8, frameon=False, title=f"Gender axis (spaCy {SPACY_MODEL})", title_fontsize=8)
ax.set_xlabel("Gender Score Decile", labelpad=34)
ax.annotate("", xy=(0.02, -0.11), xytext=(0.44, -0.11), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("", xy=(0.98, -0.11), xytext=(0.56, -0.11), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.23, -0.14, "More Feminine", ha="center", va="top", transform=ax.transAxes, fontsize=8)
ax.text(0.77, -0.14, "More Masculine", ha="center", va="top", transform=ax.transAxes, fontsize=8)
for ext in ("png", "pdf"): fig.savefig(OUT_CHAN / f"figS_chinese_replication.{ext}", dpi=300, bbox_inches="tight")
print("figure written to", OUT_CHAN / "figS_chinese_replication.png")

norms_song = load_norms(NORMS_SONG, word_col="word_chinese", conc_col="concrete_mean", sd_col=None, freq_col="frequency_subtitle", log_freq=True)
print(f"{len(norms_song):,} distinct rated words; concreteness mean {norms_song.conc.mean():.2f} (sd {norms_song.conc.std():.2f}) on a {SCALE} scale")
S2, DEC2, PS2 = run_all(norms_song, "zh_song", OUT_SONG)
display(S2.drop(columns=["language", "model"]).round(4))
display(DEC2.pivot_table(index="decile", columns="seeds", values="mean").round(3))
print(PS2[["specification", "n_pairs", "r", "ci_lo", "ci_hi", "n", "most_female", "most_male"]].round(3).to_string(index=False))

for out in (OUT_CHAN, OUT_SONG):
    print(out.name)
    for f in sorted(out.iterdir()): print("  ", f.name)
