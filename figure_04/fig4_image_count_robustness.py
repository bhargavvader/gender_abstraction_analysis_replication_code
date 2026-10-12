#!/usr/bin/env python3
"""
Fig. 4 robustness: unique objects per search term vs. (A) concreteness, (B) image gender,
(C) text gender, CONTROLLING FOR THE NUMBER OF IMAGES in each term's image set.

Inputs
  word_surprise_metrics_with_image_counts.csv   (unique_objects, total_objects, image_count, ...)
  image_class_w_conc_PROCESSED.csv              (conc, num_faces, frac_male)
  MASTER_all_models_gender_all_concreteness.csv (optional; 12-model text-gender composite for panel C)
    -> if absent, panel C falls back to the single OpenAI (GPT4) score in the metrics file
       (search_word_gender) and is labelled as provisional.

Outputs (OUT_DIR)
  fig4_men_are_many_imgcount_control.png / .pdf   three-panel figure, y adjusted for log(image count)
  fig4_imgcount_robustness_table.csv              raw vs. adjusted estimates under several specifications
  fig4_imgcount_merged_data.csv                   per-term analysis dataset
"""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from scipy import stats
import statsmodels.api as sm

# ----------------------------------------------------------------------------- CONFIG
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import data, results, IMAGE, MASTER

METRICS = data("images", "word_surprise_metrics_with_image_counts.csv", env="GC_SURPRISE_COUNTS_FILE")
IMAGES = IMAGE                         # override with GC_IMAGE_FILE
# MASTER is optional; override with GC_MASTER_FILE (see provisional fallback below).
OUT_DIR = results("figure_04_image_count_robustness")  # under GC_RESULTS_DIR, if set
MIN_FACES = 1
YLIM = (0, 1500)
CONTROL = "log_img"            # primary control: log(number of images)
MIN_MODELS = 8                 # min models per word for the composite (as in figure_04.py)
# The paper's 12-model registry (registry.py, 5 Oct 2026): display name -> master column
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
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------- LOAD + MERGE
def key(s):
    return s.astype(str).str.lower().str.strip()

w = pd.read_csv(METRICS)
img = pd.read_csv(IMAGES, low_memory=False)
w["key"], img["key"] = key(w["word"]), key(img["query"])
img = img.drop_duplicates("key")
d = w.merge(img[["key", "conc", "num_faces", "frac_male", "freq"]], on="key", how="inner")
d = d[d["image_count"] > 0].copy()
d["log_img"] = np.log(d["image_count"])
d["log_tot"] = np.log(d["total_objects"])
d["img_c"] = d["image_count"] - d["image_count"].mean()
d["img_c2"] = d["img_c"] ** 2
d["img_decile"] = pd.qcut(d["image_count"].rank(method="first"), 10, labels=False)

# ---- text gender for panel C
text_gender_label, provisional = None, False
if os.path.exists(MASTER):
    m = pd.read_csv(MASTER, low_memory=False)
    mw = m[m["dataset"] == "word"].copy()                        # the 39,953-word vocabulary
    mw["key"] = key(mw["term"]); mw = mw.drop_duplicates("key")
    cols = [c for _, c in REGISTRY]
    missing = [c for c in cols if c not in mw.columns]
    if missing:
        sys.exit(f"Master is missing registry columns {missing}; available: {[c for c in mw.columns if c.startswith('gender_')]}")
    # sign check (as in figure_04.py): every model must score male anchors above female anchors
    anch_m = ["man", "men", "male", "boy", "father", "king", "he", "him", "husband", "brother"]
    anch_f = ["woman", "women", "female", "girl", "mother", "queen", "she", "her", "wife", "sister"]
    sidx = mw.set_index("key")
    anch_m = [a for a in anch_m if a in sidx.index]; anch_f = [a for a in anch_f if a in sidx.index]
    for c in cols:
        diff = sidx.loc[anch_m, c].mean() - sidx.loc[anch_f, c].mean()
        assert diff > 0, f"{c}: male anchors not above female anchors ({diff:.3f})"
    # composite: z within the 39,953-word vocabulary (ddof=0), equal-weight mean, >= MIN_MODELS models
    Z = mw[cols].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    n_avail = Z.notna().sum(axis=1)
    mw["gender_z_mean"] = Z.mean(axis=1).where(n_avail >= MIN_MODELS)
    mw["gender_z_nmodels"] = n_avail
    d = d.merge(mw[["key", "gender_z_mean", "gender_z_nmodels"]], on="key", how="left")
    text_gender_label = f"Text-Based Gender Association\n(Mean z Across {len(REGISTRY)} Models)"
    print(f"Panel C: {len(REGISTRY)}-model registry composite from master; "
          f"{d['gender_z_mean'].notna().sum():,} of {len(d):,} image terms scored "
          f"(all 12 models: {(d['gender_z_nmodels']==12).mean():.1%})")
else:
    g = d["search_word_gender"]
    d["gender_z_mean"] = (g - g.mean()) / g.std()
    text_gender_label = "Text-Based Gender Association\n(GPT4 Score, z; Provisional)"
    provisional = True
    print("Panel C: master file not found -> provisional single-model (GPT4) score")

print(f"merged N = {len(d):,}; image_count mean {d.image_count.mean():.1f}, sd {d.image_count.std():.1f}, "
      f"median {d.image_count.median():.0f}, range {d.image_count.min()}-{d.image_count.max()}")

# ----------------------------------------------------------------------------- STATS HELPERS
def pearson_ci(r, n, k=0):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3 - k)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

def resid(y, C):
    return sm.OLS(y, sm.add_constant(C)).fit().resid

def partial(x, y, C):
    rx, ry = resid(x, C), resid(y, C)
    r, p = stats.pearsonr(rx, ry)
    rho = stats.spearmanr(rx, ry).correlation
    lo, hi = pearson_ci(r, len(x), k=C.shape[1])
    return r, lo, hi, p, rho

def fit(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit()

PANELS = [  # (id, title, x column, x label, sample filter)
    ("A", "A. Concreteness", "conc",
     "Search-Term Concreteness\n(1 = Abstract, 5 = Concrete)", lambda s: s),
    ("B", "B. Image Gender", "frac_male",
     f"Fraction of Male Faces in the Search Term's Image Set\n(Words with ≥{MIN_FACES} Face)",
     lambda s: s[s["num_faces"] >= MIN_FACES]),
    ("C", "C. Text Gender", "gender_z_mean", text_gender_label, lambda s: s),
]

SPECS = {  # label -> list of control columns (None = no control)
    "Raw (Fig. 4)":                         None,
    "+ log(images)":                        ["log_img"],
    "+ images (linear)":                    ["img_c"],
    "+ images (linear + quadratic)":        ["img_c", "img_c2"],
    "+ image-count decile fixed effects":   "decile",
    "+ log(images) + log(object tokens)":   ["log_img", "log_tot"],
}

rows = []
panel_stats = {}
for pid, title, xc, xlab, filt in PANELS:
    s = filt(d).dropna(subset=[xc, "unique_objects"]).copy()
    y = s["unique_objects"].astype(float)
    for spec, ctrl in SPECS.items():
        for samp, ss in [("all", s), ("≥200 images", s[s["image_count"] >= 200])]:
            if samp != "all" and spec not in ("Raw (Fig. 4)", "+ log(images)"):
                continue
            yy = ss["unique_objects"].astype(float); xx = ss[xc].astype(float)
            if ctrl is None:
                r, p = stats.pearsonr(xx, yy); lo, hi = pearson_ci(r, len(ss))
                rho = stats.spearmanr(xx, yy).correlation
                f = fit(yy, ss[[xc]]); slope, se = f.params[xc], f.bse[xc]
            else:
                C = pd.get_dummies(ss["img_decile"], drop_first=True, dtype=float) if ctrl == "decile" else ss[ctrl]
                r, lo, hi, p, rho = partial(xx, yy, C)
                f = fit(yy, pd.concat([ss[[xc]], C], axis=1)); slope, se = f.params[xc], f.bse[xc]
            rows.append(dict(panel=pid, predictor=xc, specification=spec, sample=samp, N=len(ss),
                             r=r, r_ci_low=lo, r_ci_high=hi, p=p, spearman_rho=rho,
                             slope=slope, slope_se=se, slope_p=f.pvalues[xc]))
            if spec == "+ log(images)" and samp == "all":
                panel_stats[pid] = dict(r=r, lo=lo, hi=hi, p=p, rho=rho, slope=slope, n=len(ss), fit=f)
            if spec == "Raw (Fig. 4)" and samp == "all":
                panel_stats[pid + "_raw"] = dict(r=r, lo=lo, hi=hi, rho=rho, slope=slope, n=len(ss))

tab = pd.DataFrame(rows)
PUBLISHED = {"A": (-0.381, -0.376, -83.5, 36124), "B": (0.113, 0.080, 131.8, 35603), "C": (0.074, 0.099, 30.1, 36124)}
print("\nCheck against published Fig. 4 (r, rho, slope, N):")
for pid, (r0, rho0, b0, n0) in PUBLISHED.items():
    q = panel_stats[pid + "_raw"]
    ok = abs(q["r"] - r0) < 0.0015 and abs(q["rho"] - rho0) < 0.0015 and abs(q["slope"] - b0) < 0.1 and q["n"] == n0
    print(f"  {pid}: computed r={q['r']:+.3f} rho={q['rho']:+.3f} slope={q['slope']:+.1f} N={q['n']:,}  "
          f"published r={r0:+.3f} rho={rho0:+.3f} slope={b0:+.1f} N={n0:,}  -> {'MATCH' if ok else 'MISMATCH'}")
tab.to_csv(f"{OUT_DIR}/fig4_imgcount_robustness_table.csv", index=False)
d.to_csv(f"{OUT_DIR}/fig4_imgcount_merged_data.csv", index=False)

# ----------------------------------------------------------------------------- FIGURE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=False)
for ax, (pid, title, xc, xlab, filt) in zip(axes, PANELS):
    s = filt(d).dropna(subset=[xc, "unique_objects"]).copy()
    ps = panel_stats[pid]; f = ps["fit"]
    b_img, mean_log = f.params[CONTROL], s[CONTROL].mean()
    # partial-residual adjustment: y evaluated as if every term had the mean (log) number of images
    y_adj = s["unique_objects"] - b_img * (s[CONTROL] - mean_log)
    x = s[xc].astype(float)
    ax.scatter(x, y_adj, s=1.2, color="#4f81bd", alpha=0.18, linewidths=0, rasterized=True)
    # fitted line + 95% CI from the multiple regression at the mean control value
    xs = np.linspace(x.min(), x.max(), 200)
    X = sm.add_constant(pd.DataFrame({xc: xs, CONTROL: mean_log}), has_constant="add")
    pred = f.get_prediction(X).summary_frame(alpha=0.05)
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color="#d62728", alpha=0.25, linewidths=0)
    ax.plot(xs, pred["mean"], color="#d62728", lw=2.2, label="OLS fit, adjusted for number of images")
    ptxt = "p < 0.001" if ps["p"] < 0.001 else f"p = {ps['p']:.3f}"
    inset = (f"Partial Pearson r = {ps['r']:+.3f}  [95% CI {ps['lo']:+.3f}, {ps['hi']:+.3f}], {ptxt}\n"
             f"Partial Spearman ρ = {ps['rho']:+.3f}\n"
             f"adjusted slope = {ps['slope']:+.1f} objects per unit x\n"
             f"(unadjusted: r = {panel_stats[pid+'_raw']['r']:+.3f}, slope = {panel_stats[pid+'_raw']['slope']:+.1f})\n"
             f"N = {ps['n']:,} search terms")
    if pid == "C" and provisional:
        inset += "\nProvisional: GPT4 score only (12-model composite pending)"
    ax.text(0.02, 0.98, inset, transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95))
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=12.5)
    ax.set_ylim(*YLIM)
    if pid == "A":
        ax.set_ylabel("Unique Objects Detected in Search Term's Image Set\n(Adjusted for Number of Images)", fontsize=12.5)
        ax.set_xlim(0.9, 5.1)
    elif pid == "B":
        ax.set_xlim(-0.02, 1.02)
    else:
        ax.set_xlim(-3, 3)
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes); ya = -0.125
        for x0, x1, lab in [(-0.25, -2.9, "More Female"), (0.25, 2.9, "More Male")]:
            ax.annotate("", xy=(x1, ya), xytext=(x0, ya), xycoords=tr, textcoords=tr,
                        arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4, shrinkA=0, shrinkB=0),
                        annotation_clip=False)
            ax.text((x0 + x1) / 2, ya + 0.022, lab, transform=tr, ha="center", va="bottom", fontsize=11.5, color="0.3")
        ax.xaxis.set_label_coords(0.5, -0.19)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
fig.tight_layout(w_pad=2.5)
fig.savefig(f"{OUT_DIR}/fig4_men_are_many_imgcount_control.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig4_men_are_many_imgcount_control.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------------- CONSOLE SUMMARY
pd.set_option("display.width", 200)
show = tab.copy()
show["r [95% CI]"] = show.apply(lambda r: f"{r.r:+.3f} [{r.r_ci_low:+.3f}, {r.r_ci_high:+.3f}]", axis=1)
print(show[["panel", "specification", "sample", "N", "r [95% CI]", "spearman_rho", "slope", "slope_se"]]
      .round({"spearman_rho": 3, "slope": 1, "slope_se": 1}).to_string(index=False))
