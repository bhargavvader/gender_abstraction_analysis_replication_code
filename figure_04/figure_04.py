"""figure_04.py  (= unique_objects_analysis.py, final version of 21 Sept 2026)

Unique detected objects per search term vs. concreteness, image gender, text gender (fig4_men_are_many).

Inputs
  word_surprise_metrics.csv           (visual-divergence interface v1; unique_objects; from the images-objects
                                       artifact store, results/metrics/ -- 38,883 rows x 16 columns)
  image_class_w_conc_PROCESSED.csv    (face classification per query; frac_male, num_faces, conc)
  Concreteness_ratings_Brysbaert...   (Conc.M, authoritative concreteness)
  MASTER_all_models_gender_all_concreteness.csv (text-model gender scores, dataset=='word')

Outputs (results/figure_04/): fig_combined_unique_objects.png (= fig4_men_are_many), the three standalone panels,
  unique_objects_stats.csv (main + robustness rows), unique_objects_analysis_dataset.csv (merged per-word data).

Panel c averages the 12-model registry of registry.py (shared with Figs. 1, 2 and 5), each model z-scored within
the 39,953-word vocabulary.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MASTER, BRYS, IMAGE, SURPRISE, results
from registry import REGISTRY

OUT = results("figure_04")

# ----------------------------------------------------------------------------- CONFIG
MIN_FACES = 1                 # words with at least this many detected faces enter panel 2
MIN_MODELS = 8                # min number of gender models with a score for a word's composite
N_BINS = 10                   # decile bins (used only for the panel-specific binning options below)

# text-based gender composite: the 12-model registry shared with Figs. 1, 2 and 5
PRIMARY_MODELS = [c for _, c in REGISTRY]
N_PRIMARY = len(PRIMARY_MODELS)

# ----------------------------------------------------------------------------- LOAD
w = pd.read_csv(SURPRISE)
img = pd.read_csv(IMAGE, low_memory=False)
bry = pd.read_csv(BRYS)
master = pd.read_csv(MASTER, low_memory=False)
for d, c in [(w, "word"), (img, "query"), (bry, "Word"), (master, "term")]:
    d["key"] = d[c].astype(str).str.lower().str.strip()
img = img.drop_duplicates("key")
bry = bry.drop_duplicates("key")
mw = master[master["dataset"] == "word"].drop_duplicates("key").copy()

# ----------------------------------------------------------------------------- GENDER COMPOSITE
# sign check: every model must score male anchors above female anchors
anch_m = ["man", "men", "male", "boy", "father", "king", "he", "him", "husband", "brother"]
anch_f = ["woman", "women", "female", "girl", "mother", "queen", "she", "her", "wife", "sister"]
s = mw.set_index("key")
anch_m = [a for a in anch_m if a in s.index]
anch_f = [a for a in anch_f if a in s.index]
for c in PRIMARY_MODELS:
    diff = s.loc[anch_m, c].mean() - s.loc[anch_f, c].mean()
    assert diff > 0, f"{c}: male anchors not above female anchors ({diff:.3f})"

def composite(cols, name):
    z = mw[cols].apply(lambda col: (col - col.mean()) / col.std(ddof=0))  # z within the 39,953-word vocab
    n_avail = z.notna().sum(axis=1)
    mw[name] = z.mean(axis=1).where(n_avail >= MIN_MODELS)
    mw[name + "_nmodels"] = n_avail

composite(PRIMARY_MODELS, "gender_z_mean")            # the 12-model registry composite

# ----------------------------------------------------------------------------- MERGE
df = (w[["key", "word", "unique_objects", "total_objects", "object_entropy", "search_word_gender"]]
      .merge(img[["key", "num_faces", "num_males", "num_females", "frac_male"]], on="key")
      .merge(bry[["key", "Conc.M"]].rename(columns={"Conc.M": "concreteness"}), on="key", how="left")
      .merge(mw[["key", "gender_z_mean", "gender_z_mean_nmodels"]], on="key", how="left"))
df.loc[df["num_faces"] < MIN_FACES, "frac_male"] = np.nan   # frac_male undefined without faces
df.to_csv(f"{OUT}/unique_objects_analysis_dataset.csv", index=False)

# ----------------------------------------------------------------------------- STATS
def corr_stats(x, y):
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    n = len(x)
    r, p = stats.pearsonr(x, y)
    zf = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(zf - 1.96 * se), np.tanh(zf + 1.96 * se)
    rho, p_rho = stats.spearmanr(x, y)
    ols = stats.linregress(x, y)
    return dict(n=n, r=r, r_lo=lo, r_hi=hi, p=p, rho=rho, p_rho=p_rho,
                slope=ols.slope, intercept=ols.intercept, slope_se=ols.stderr,
                beta_std=r)  # standardized slope == r for simple OLS

panels = [
    ("concreteness", "Search-Term Concreteness\n(1 = Abstract, 5 = Concrete)",
     "A. Concreteness"),
    ("frac_male", f"Fraction of Male Faces in the Search Term's Image Set\n(Words with ≥{MIN_FACES} Face)",
     "B. Image Gender"),
    ("gender_z_mean", f"Text-Based Gender Association\n(Mean z Across {N_PRIMARY} Models)",
     "C. Text Gender"),
]
results_ = {}
for col, _, _ in panels:
    results_[col] = corr_stats(df[col], df["unique_objects"])

# robustness rows
rob = []
for t in [1, 5, 10, 20, 50]:
    sub = df[df["num_faces"] >= t]
    st = corr_stats(sub["frac_male"], sub["unique_objects"])
    rob.append(dict(analysis=f"frac_male, num_faces>={t}", **st))
for col, lab in [("gender_z_mean", f"text gender: {N_PRIMARY}-model registry composite (primary)"),
                 ("search_word_gender", "text gender: single-model (GPT4) score embedded in metrics file")]:
    st = corr_stats(df[col], df["unique_objects"])
    rob.append(dict(analysis=lab, **st))
# log-outcome robustness
for col, _, _ in panels:
    st = corr_stats(df[col], np.log(df["unique_objects"]))
    rob.append(dict(analysis=f"{col} vs log(unique_objects)", **st))
# partial: text gender controlling for concreteness
sub = df[["gender_z_mean", "concreteness", "unique_objects"]].dropna()
X = np.column_stack([np.ones(len(sub)), sub["concreteness"], sub["gender_z_mean"]])
beta, *_ = np.linalg.lstsq(X, sub["unique_objects"], rcond=None)
resid_y = sub["unique_objects"] - X[:, :2] @ np.linalg.lstsq(X[:, :2], sub["unique_objects"], rcond=None)[0]
resid_x = sub["gender_z_mean"] - X[:, :2] @ np.linalg.lstsq(X[:, :2], sub["gender_z_mean"], rcond=None)[0]
pr, pp = stats.pearsonr(resid_x, resid_y)
rob.append(dict(analysis="text gender | concreteness (partial r)", n=len(sub), r=pr, p=pp,
                slope=beta[2]))
sub2 = df[["frac_male", "concreteness", "unique_objects"]].dropna()
X = np.column_stack([np.ones(len(sub2)), sub2["concreteness"], sub2["frac_male"]])
beta2, *_ = np.linalg.lstsq(X, sub2["unique_objects"], rcond=None)
ry = sub2["unique_objects"] - X[:, :2] @ np.linalg.lstsq(X[:, :2], sub2["unique_objects"], rcond=None)[0]
rx = sub2["frac_male"] - X[:, :2] @ np.linalg.lstsq(X[:, :2], sub2["frac_male"], rcond=None)[0]
pr2, pp2 = stats.pearsonr(rx, ry)
rob.append(dict(analysis="image gender | concreteness (partial r)", n=len(sub2), r=pr2, p=pp2,
                slope=beta2[2]))

main_tbl = pd.DataFrame([dict(analysis=f"{c} vs unique_objects (primary)", **results_[c]) for c, _, _ in panels])
pd.concat([main_tbl, pd.DataFrame(rob)]).to_csv(f"{OUT}/unique_objects_stats.csv", index=False)

# ----------------------------------------------------------------------------- PLOTS
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False})
COL_PT, COL_FIT, COL_BIN = "#7a9cc6", "#c0392b", "#111111"

PANEL_OPTS = {
    "concreteness": dict(bins="decile", xlim=None),
    "frac_male": dict(bins="width", xlim=(-0.02, 1.02)),        # equal-width 0.1 bins: deciles collapse (IQR 0.56-0.77)
    "gender_z_mean": dict(bins="decile", xlim=(-3, 3), arrows=True),          # central ~99.5% of terms; tails are seed-like words
}

def draw_panel(ax, col, xlabel, title, show_ylabel=True):
    opts = PANEL_OPTS[col]
    d = df[[col, "unique_objects"]].dropna()
    x, y = d[col].to_numpy(), d["unique_objects"].to_numpy()
    st = results_[col]
    ax.scatter(x, y, s=3, alpha=0.12, color=COL_PT, linewidths=0, rasterized=True)
    # OLS fit with 95% CI band
    xs = np.linspace(x.min(), x.max(), 200)
    yhat = st["intercept"] + st["slope"] * xs
    n = len(x); resid = y - (st["intercept"] + st["slope"] * x)
    s_err = np.sqrt((resid ** 2).sum() / (n - 2))
    se_fit = s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    ax.plot(xs, yhat, color=COL_FIT, lw=2, zorder=5, label="OLS fit")
    ax.fill_between(xs, yhat - 1.96 * se_fit, yhat + 1.96 * se_fit, color=COL_FIT, alpha=0.2, zorder=3)
    ptxt = "p < 0.001" if st["p"] < 0.001 else f"p = {st['p']:.3f}"
    ax.text(0.02, 0.97,
            f"Pearson r = {st['r']:+.3f}  [95% CI {st['r_lo']:+.3f}, {st['r_hi']:+.3f}], {ptxt}\n"
            f"Spearman ρ = {st['rho']:+.3f}\n"
            f"slope = {st['slope']:+.1f} objects per unit x\nN = {st['n']:,} search terms",
            transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.95), zorder=6)
    if opts.get("arrows"):
        import matplotlib.transforms as mtrans
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        ya = -0.105
        for tip, txt, lab in [(-2.95, -1.05, "More Female"), (2.95, 1.05, "More Male")]:
            ax.annotate(lab, xy=(tip, ya), xytext=(txt, ya), xycoords=tr, textcoords=tr,
                        ha="center", va="center", fontsize=11.5, color="0.25",
                        arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4, shrinkB=0),
                        annotation_clip=False)
        ax.set_xlabel(xlabel, fontsize=12.5, labelpad=36)
    else:
        ax.set_xlabel(xlabel, fontsize=12.5)
    if show_ylabel:
        ax.set_ylabel("Unique Objects Detected in Search Term's Image Set", fontsize=12.5)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1500)
    if opts["xlim"] is not None:
        ax.set_xlim(*opts["xlim"])
    ax.legend(loc="lower right", fontsize=11, frameon=True)

# individual figures
for i, (col, xlabel, title) in enumerate(panels, 1):
    fig, ax = plt.subplots(figsize=(7.6, 6.3))
    draw_panel(ax, col, xlabel, title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig{i}_{col}_vs_unique_objects.png", dpi=200)
    plt.close(fig)

# combined (= fig4_men_are_many)
fig, axes = plt.subplots(1, 3, figsize=(20, 6.7), sharey=True)
for ax, (col, xlabel, title) in zip(axes, panels):
    draw_panel(ax, col, xlabel, title, show_ylabel=(ax is axes[0]))
fig.tight_layout(w_pad=3)
fig.savefig(f"{OUT}/fig_combined_unique_objects.png", dpi=200)
fig.savefig(f"{OUT}/fig4_men_are_many.png", dpi=200)
fig.savefig(f"{OUT}/fig4_men_are_many.pdf")

# ----------------------------------------------------------------------------- REPORT
pd.set_option("display.width", 200)
print(main_tbl[["analysis", "n", "r", "r_lo", "r_hi", "p", "rho", "slope"]].to_string(index=False))
print()
print(pd.DataFrame(rob)[["analysis", "n", "r", "p", "slope"]].to_string(index=False))
print("\ngender composite coverage (n models):")
print(df["gender_z_mean_nmodels"].value_counts().sort_index().to_string())
print("composite correlations:")
print(df[["gender_z_mean", "search_word_gender", "frac_male"]].corr().round(3).to_string())
