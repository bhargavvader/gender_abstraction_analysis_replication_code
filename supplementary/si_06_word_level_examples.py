"""
==============================================================================
si_06_word_level_examples.py

SUPPLEMENTARY SECTION 6 / SUPPLEMENTARY FIG. 1 / SUPPLEMENTARY TABLE 9
Reproduction script: by-domain gender x concreteness analysis + 3x3 figure
------------------------------------------------------------------------------
Reproduces, from raw inputs:
  1. The POS-filtered WordNet domain map (verb.communication, adv.all,
     adj.all, noun.artifact) for the Brysbaert 40k words (loaded from the
     tracked file when present).
  2. The Model x Domain correlation table (Pearson r, 95% CI, BH-FDR over the
     48-cell family) for the 12 language models of the registry.
  3. The final 3x3 panel figure (rows: Qwen3, GPT4, Nomic v2 MoE;
     columns: Acts of Communication, Adjectives (General),
     Nouns (Artifacts & Ideas)) with all agreed styling decisions.

Inputs (set paths in CONFIG below):
  - MASTER_all_models_gender_all_concreteness.csv (the full master; 'word' rows,
    12 registry gender columns)
  - Concreteness_ratings_Brysbaert_et_al_BRM(_ext).csv  (Brysbaert 40k norms,
    must include Word, Conc.M, SUBTLEX, Dom_Pos columns)
  - supplementary/domain_labels_40k_wordnet_POSfiltered.csv (tracked in the repo:
    the POS-filtered WordNet domain map, frozen against WordNet version changes;
    counts verb.communication 1,393 / adj.all 5,024 / noun.artifact 4,634 /
    adv.all 1,654). If absent, it is rebuilt from WordNet (requires `nltk`).

Outputs (written to results/si_06/):
  - BY_DOMAIN_correlations_final_models_POSfiltered.csv   (Supplementary Table 9 + the adv.all cells)
  - figS_domain_word_examples.png                          (Supplementary Fig. 1)

Dependencies: pandas, numpy, scipy >= 1.11 (false_discovery_control), matplotlib
(+ nltk only if the domain map needs to be rebuilt).

Method notes (documenting the decisions accumulated in the analysis):
  * Domain assignment: a word joins a domain iff WordNet lists a sense of it
    under that lexname AND the word's dominant POS in Brysbaert (Dom_Pos)
    matches the domain's POS (verb.* -> Verb, noun.* -> Noun, adj.all ->
    Adjective [WordNet adj.ppl folded in], adv.all -> Adverb). Multiword
    entries lack a usable Dom_Pos and are admitted only to noun.artifact,
    and only when WordNet lists them exclusively as nouns. This prevents
    rare cross-POS senses (e.g. 'butterfly'/'skirt' as communication verbs)
    from contaminating domains.
  * Correlations use all words in each Model x Domain cell; FDR (BH) is
    applied across all 48 cells (12 models x 4 domains) even though the
    figure shows 9 of them.
  * Highlighted examples: within each panel, the 3 most salient
    female-and-concrete words (red) and 3 most salient male-and-abstract
    words (blue), salience = -z(gender) * z(concreteness); a per-panel
    exclusion list holds hand-curated swaps agreed during figure revision
    (excluded words remain in the data; only the highlight moves to the
    next-ranked candidate). GPT4/noun.artifact labels single words only.
  * Per-panel y-axis cutoffs crop a small number of extreme male-end points
    (counts are printed at run time for figure captions); labeled examples
    are always chosen from below the cutoff.
  * Label placement is deterministic: most-constrained-first greedy search
    over candidate offsets around each dot, accepting the first position
    fully inside the panel frame, clear of every highlighted dot, and clear
    of other labels; the script verifies all three constraints and reports.
==============================================================================
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import BRYS, MASTER, results
from registry import REGISTRY

# =============================================================================
# CONFIG
# =============================================================================
CONC_FILE   = BRYS                                           # the _ext file has Word, Conc.M, SUBTLEX, Dom_Pos
OUT_DIR     = results("si_06")
DOMAIN_FILE = os.environ.get("GC_DOMAIN_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "domain_labels_40k_wordnet_POSfiltered.csv"))   # tracked in the repo; rebuilt if absent
FIG_NAME    = "figS_domain_word_examples.png"

# gender-score column in the master -> display name (all 12 registry models enter the correlation table)
MODEL_LABELS = {col: name for name, col in REGISTRY}

# figure layout
FIG_MODELS  = ["Qwen3", "GPT4", "Nomic v2 MoE"]            # rows, top to bottom
FIG_DOMAINS = ["verb.communication", "adj.all", "noun.artifact"]  # columns
DOMAIN_TITLES = {
    "verb.communication": "Acts of Communication",
    "adj.all":            "Adjectives (General)",
    "noun.artifact":      "Nouns (Artifacts & Ideas)",
}
ALL_DOMAINS = ["verb.communication", "adv.all", "adj.all", "noun.artifact"]

BLUE, RED = "#2166AC", "#B2182B"   # male-abstract / female-concrete
N_PER_CORNER = 3

# per-panel y-axis limits (lo, hi); None = data range + 2% pad
YLIMS = {
    ("Nomic v2 MoE", "verb.communication"): (None, 0.06),
    ("Nomic v2 MoE", "adj.all"):            (None, 0.13),
    ("GPT4",         "noun.artifact"):      (-0.2, 0.12),
    ("Nomic v2 MoE", "noun.artifact"):      (None, 0.08),
}
# hand-curated highlight exclusions (words stay in the data)
EXCLUDE = {
    ("Nomic v2 MoE", "noun.artifact"):      {"superman"},
    ("GPT4",         "verb.communication"): {"embroider"},
    ("Qwen3",        "verb.communication"): {"embroider"},
    ("Qwen3",        "adj.all"):            {"goddamn"},
    ("GPT4",         "noun.artifact"):      {"max", "hypo", "panache",
                                             "longshot", "override", "dandy"},
}
SINGLE_WORD_ONLY = {("GPT4", "noun.artifact")}

# =============================================================================
# STEP 1 -- domain map (load or rebuild from WordNet)
# =============================================================================
def build_domain_map(conc: pd.DataFrame, words) -> pd.DataFrame:
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet")
        from nltk.corpus import wordnet as wn

    targets = {"verb.communication": "v", "adv.all": "r",
               "adj.all": "a", "noun.artifact": "n"}
    dompos2wn = {"Noun": "n", "Verb": "v", "Adjective": "a", "Adverb": "r"}
    pos_map = dict(zip(conc["Word"], conc["Dom_Pos"]))

    rows = []
    for w in words:
        syns = wn.synsets(w.replace(" ", "_"))
        if not syns:
            continue
        lex = {("adj.all" if s.lexname() == "adj.ppl" else s.lexname())
               for s in syns}
        wn_pos = {("a" if s.pos() == "s" else s.pos()) for s in syns}
        dp = dompos2wn.get(pos_map.get(w))
        for dmn, req in targets.items():
            if dmn not in lex:
                continue
            ok = (dp == req) if dp is not None else (req == "n" and wn_pos == {"n"})
            if ok:
                rows.append((w, dmn))
    return pd.DataFrame(rows, columns=["Word", "domain"]).drop_duplicates()


gender_raw = pd.read_csv(MASTER, low_memory=False, keep_default_na=False, na_values=["", "NA", "NaN"])
gender_raw = gender_raw[gender_raw["dataset"] == "word"].drop_duplicates("term")     # single-word rows of the master
for c in MODEL_LABELS: gender_raw[c] = pd.to_numeric(gender_raw[c], errors="coerce")
conc = pd.read_csv(CONC_FILE)

if os.path.exists(DOMAIN_FILE):
    dom = pd.read_csv(DOMAIN_FILE)
    print(f"Loaded domain map: {DOMAIN_FILE}")
else:
    print("Domain map not found -- rebuilding from WordNet (needs nltk)...")
    dom = build_domain_map(conc, gender_raw["term"].tolist())
    dom.to_csv(DOMAIN_FILE, index=False)
print(dom["domain"].value_counts().to_string(), "\n")

# =============================================================================
# STEP 2 -- merge and correlation table (all 9 models x 4 domains)
# =============================================================================
long = (gender_raw[["term"] + list(MODEL_LABELS)]
        .melt("term", var_name="col", value_name="gender").dropna())
long["Model"] = long["col"].map(MODEL_LABELS)
dat = (long.merge(conc[["Word", "Conc.M", "SUBTLEX"]]
                  .rename(columns={"Word": "term", "Conc.M": "concreteness"}),
                  on="term")
           .merge(dom.rename(columns={"Word": "term"}), on="term"))

rows = []
for (m, dmn), sub in dat.groupby(["Model", "domain"]):
    r, p = stats.pearsonr(sub["gender"], sub["concreteness"])
    n = len(sub)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    rows.append(dict(Model=m, domain=dmn, N=n, r=r,
                     ci_lo=np.tanh(z - 1.96 * se),
                     ci_hi=np.tanh(z + 1.96 * se), p=p))
cor = pd.DataFrame(rows)
cor["p_fdr"] = stats.false_discovery_control(cor["p"], method="bh")
cor.to_csv(os.path.join(
    OUT_DIR, "BY_DOMAIN_correlations_final_models_POSfiltered.csv"), index=False)
print(cor.pivot(index="domain", columns="Model", values="r").round(3), "\n")

# =============================================================================
# STEP 3 -- figure
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
panels = []
for i, m in enumerate(FIG_MODELS):
    for j, dmn in enumerate(FIG_DOMAINS):
        ax = axes[i, j]
        sub = dat[(dat["Model"] == m) & (dat["domain"] == dmn)].copy()
        sub["zg"] = stats.zscore(sub["gender"])
        sub["zc"] = stats.zscore(sub["concreteness"])
        sub["sal"] = -sub["zg"] * sub["zc"]

        ax.scatter(sub["concreteness"], sub["gender"],
                   s=7, c="0.75", alpha=.35, lw=0)
        b1, b0, *_ = stats.linregress(sub["concreteness"], sub["gender"])
        xs = np.linspace(sub["concreteness"].min(), sub["concreteness"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, "k-", lw=1.6)

        lo, hi = YLIMS.get((m, dmn), (None, None))
        pool = sub[~sub["term"].isin(EXCLUDE.get((m, dmn), set()))]
        if (m, dmn) in SINGLE_WORD_ONLY:
            pool = pool[~pool["term"].str.contains(" ")]
        if hi is not None:
            pool = pool[pool["gender"] <= hi]
        fc = pool[(pool["zg"] < 0) & (pool["zc"] > 0)].nlargest(N_PER_CORNER, "sal")
        ma = pool[(pool["zg"] > 0) & (pool["zc"] < 0)].nlargest(N_PER_CORNER, "sal")

        items = []
        for top, col in [(ma, BLUE), (fc, RED)]:
            ax.scatter(top["concreteness"], top["gender"], s=34, c=col, zorder=3)
            for _, row in top.iterrows():
                items.append((row["term"], row["concreteness"], row["gender"], col))

        xr = sub["concreteness"].max() - sub["concreteness"].min()
        yr = sub["gender"].max() - sub["gender"].min()
        ax.set_xlim(sub["concreteness"].min() - .02 * xr,
                    sub["concreteness"].max() + .02 * xr)
        ax.set_ylim(sub["gender"].min() - .02 * yr if lo is None else lo,
                    sub["gender"].max() + .02 * yr if hi is None else hi)
        if hi is not None or lo is not None:
            n_crop = int(((sub["gender"] > (hi if hi is not None else np.inf)) |
                          (sub["gender"] < (lo if lo is not None else -np.inf))).sum())
            print(f"  cropped points {m} / {dmn}: {n_crop} of {len(sub)}")

        st = cor[(cor["Model"] == m) & (cor["domain"] == dmn)].iloc[0]
        star = "*" if st["p_fdr"] < .05 else ""
        ax.text(.03, .04, f"r = {st['r']:.2f}{star}, N = {st['N']:,}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=10.5)
        if i == 0:
            ax.set_title(DOMAIN_TITLES[dmn], fontsize=14)
        ax.set_ylabel(f"{m}\n\nAvg. gender association\n(higher = male)"
                      if j == 0 else "", fontsize=11)
        ax.set_xlabel("Concreteness (Brysbaert Conc.M)" if i == 2 else "",
                      fontsize=11)
        panels.append((ax, items))

fig.tight_layout()
fig.canvas.draw()
ren = fig.canvas.get_renderer()

# ---- deterministic label placement -----------------------------------------
DOT_R, PAD, GAP = 7.0, 3.0, 6
CANDS = []
for dist in (GAP + 4, GAP + 14, GAP + 26, GAP + 40):
    CANDS += [( dist, 0, "left", "center"), (-dist, 0, "right", "center"),
              (0,  dist + 1, "center", "bottom"), (0, -dist - 1, "center", "top"),
              ( dist - 2,  dist - 2, "left",  "bottom"),
              (-dist + 2,  dist - 2, "right", "bottom"),
              ( dist - 2, -dist + 2, "left",  "top"),
              (-dist + 2, -dist + 2, "right", "top")]

def place_labels(ax, items):
    axbb = ax.get_window_extent()
    pts_px = ax.transData.transform([(x, y) for _, x, y, _ in items])
    order = sorted(range(len(items)),
                   key=lambda k: min(pts_px[k][0] - axbb.x0, axbb.x1 - pts_px[k][0],
                                     pts_px[k][1] - axbb.y0, axbb.y1 - pts_px[k][1]))
    inv = ax.transData.inverted()
    placed, texts = [], [None] * len(items)
    for k in order:
        term, _, _, col = items[k]
        px, py = pts_px[k]
        for dx, dy, ha, va in CANDS:
            t = ax.text(*inv.transform((px + dx, py + dy)), term,
                        color=col, fontsize=9.5, ha=ha, va=va)
            bb = t.get_window_extent(renderer=ren)
            ok = (bb.x0 >= axbb.x0 + PAD and bb.x1 <= axbb.x1 - PAD and
                  bb.y0 >= axbb.y0 + PAD and bb.y1 <= axbb.y1 - PAD)
            if ok:
                ok = not any((bb.x0 - DOT_R) < qx < (bb.x1 + DOT_R) and
                             (bb.y0 - DOT_R) < qy < (bb.y1 + DOT_R)
                             for qx, qy in pts_px)
            if ok:
                ok = all(bb.x1 < pb.x0 - 2 or bb.x0 > pb.x1 + 2 or
                         bb.y1 < pb.y0 - 2 or bb.y0 > pb.y1 + 2 for pb in placed)
            if ok:
                placed.append(bb)
                texts[k] = t
                break
            t.remove()
        if texts[k] is None:                       # last-resort fallback
            t = ax.text(*inv.transform((px - GAP - 4, py)), term,
                        color=col, fontsize=9.5, ha="right", va="center")
            texts[k] = t
            placed.append(t.get_window_extent(renderer=ren))
            print(f"  WARNING: fallback placement used for '{term}'")
    return texts

all_texts = [(ax, place_labels(ax, items), items) for ax, items in panels]

# ---- verification -----------------------------------------------------------
fig.canvas.draw()
frame_bad, dot_bad, tt_bad = [], [], []
for ax, texts, items in all_texts:
    axbb = ax.get_window_extent()
    pts_px = ax.transData.transform([(x, y) for _, x, y, _ in items])
    bbs = [t.get_window_extent(renderer=ren) for t in texts]
    for t, bb in zip(texts, bbs):
        if (bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or
                bb.y0 < axbb.y0 or bb.y1 > axbb.y1):
            frame_bad.append(t.get_text())
        for qx, qy in pts_px:
            if (bb.x0 - 4.6) < qx < (bb.x1 + 4.6) and \
               (bb.y0 - 4.6) < qy < (bb.y1 + 4.6):
                dot_bad.append(t.get_text())
    for a in range(len(bbs)):
        for b in range(a + 1, len(bbs)):
            A, B = bbs[a], bbs[b]
            if not (A.x1 < B.x0 or A.x0 > B.x1 or A.y1 < B.y0 or A.y0 > B.y1):
                tt_bad.append((texts[a].get_text(), texts[b].get_text()))
print("frame violations:", frame_bad or "NONE")
print("dot overlaps:", sorted(set(dot_bad)) or "NONE")
print("text-text overlaps:", tt_bad or "NONE")

fig.savefig(os.path.join(OUT_DIR, FIG_NAME), dpi=200)
print("figure saved:", os.path.join(OUT_DIR, FIG_NAME))
