#!/usr/bin/env python3
# ==============================================================================
# figure_03_three_panel.py   (14 Oct 2026)
#
# Fig. 3 with a new first panel:
#   A  P(Human Face in Google Image) -- the fraction of a search term's images in
#      which a face (of either gender) was detected -- by human judgements of the
#      term's concreteness.
#   B  P(Male Face in Google Image) by concreteness          (previous panel A)
#   C  P(Male Face in Google Image) by perceptual strength   (previous panel B)
#
# Panels B and C are the recipe of figure_03.R (all terms; terms without faces at
# zero; 20 equal-count bins; loess through the bin means with its 95% band; 0.5
# line). Panel A uses the same recipe on the terms whose number of images is
# known (word_surprise_metrics_with_image_counts.csv; two-word terms are keyed
# with underscores there): y = number of faces / number of images per term,
# averaged within each bin. The loess and its band are a port of R's
# stats::loess / ggplot2::geom_smooth defaults, so the panels match the R figure.
#
# PANEL_A_STYLE "quartiles" draws panel A as four bars (mean P(face) within each
# concreteness quartile, 95% CI of the mean); "loess" draws it as panels B and C.
# VOCAB_A selects panel A's vocabulary: "all" (every term with an image count,
# matching panels B and C) or "single" (single words only). Both versions are
# written; FIGURE_STEM names the "all" version and the other gets the suffix
# "_singlewords". SCALE multiplies every font, line and marker size (1.0 = the
# sizes of the two-panel R figure; a three-panel row is scaled further down on
# the page, so 1.3 keeps the final type size about where the two-panel figure's was).
#
# Inputs : data/images/image_class_w_conc_PROCESSED.csv
#          data/images/word_surprise_metrics_with_image_counts.csv
#          data/norms/lancaster_sensorimotor_norms_2019.csv
# Outputs: figures/fig3_image_results_3panel.{png,pdf} (+ _singlewords)
#          figures/fig3_image_results_3panel_bins.csv, _stats.json
# ==============================================================================
import os, json, warnings, logging
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
warnings.filterwarnings("ignore"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ================================ CONFIG ======================================
HERE          = os.path.dirname(os.path.abspath(__file__))
REPO          = os.environ.get("GC_REPO_DIR", os.path.dirname(HERE))
IMAGE_FILE    = os.environ.get("GC_IMAGE_FILE",    os.path.join(REPO, "data/images/image_class_w_conc_PROCESSED.csv"))
IMGCOUNT_FILE = os.environ.get("GC_IMGCOUNT_FILE", os.path.join(REPO, "data/images/word_surprise_metrics_with_image_counts.csv"))
LANC_FILE     = os.environ.get("GC_LANC_FILE",     os.path.join(REPO, "data/norms/lancaster_sensorimotor_norms_2019.csv"))
OUT_DIR       = os.environ.get("GC_FIG_DIR",       os.path.join(REPO, "figures"))
FIGURE_STEM   = "fig3_image_results_3panel"
PANEL_A_STYLE = "quartiles"   # "quartiles": four equal-count bins of concreteness drawn as bars with 95% CIs;
                              # "loess": 20 bins with a loess fit, as panels B and C
PANEL_A_YLIM  = (0.08, 0.14)  # y-axis of the quartile panel; None = from zero to 1.55 x the highest bar

N_BINS         = 20
SCALE          = 1.2
N_IMAGES_LABEL = "N \u2248 12 million images"      # all panels: as in figure_03.R (whole image set)
PERCEPTUAL     = ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"]
PANEL_W, PANEL_H, DPI = 7.0, 8.4, 400
LOESS_COL, BAND_COL = "#3366FF", "#999999"
# ============================== END CONFIG ====================================
os.makedirs(OUT_DIR, exist_ok=True)
FS = {k: v * SCALE for k, v in dict(tick=20, title=22, tag=32, inset=17.5).items()}
plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 18 * SCALE,
                     "axes.linewidth": 1.2 * SCALE, "xtick.major.width": SCALE, "ytick.major.width": SCALE,
                     "xtick.major.size": 5 * SCALE, "ytick.major.size": 5 * SCALE, "pdf.fonttype": 42})

# ------------------------------------------------------------------ data ------------------------------------------
im = pd.read_csv(IMAGE_FILE, low_memory=False, keep_default_na=False,
                 na_values={"Dom_Pos": ["", "NA"], "frac_males": ["NA", ""], "frac_females": ["NA", ""], "avg_age": ["NA", ""]})
for c in ["num_faces", "num_males", "num_females", "frac_male", "conc"]:
    im[c] = pd.to_numeric(im[c])
im["key"] = im["query"].str.lower().str.strip()
lan = pd.read_csv(LANC_FILE, keep_default_na=False)
for c in PERCEPTUAL:
    lan[c] = pd.to_numeric(lan[c])
lan["perc"] = lan[PERCEPTUAL].mean(axis=1); lan["key"] = lan.Word.str.lower().str.strip()
ic = pd.read_csv(IMGCOUNT_FILE, keep_default_na=False, usecols=["word", "image_count"])
ic["key"] = ic.word.astype(str).str.lower().str.strip().str.replace("_", " ", regex=False)
ic["image_count"] = pd.to_numeric(ic.image_count)
d = (im.merge(lan[["key", "perc"]].drop_duplicates("key"), on="key", how="left")
       .merge(ic[["key", "image_count"]].drop_duplicates("key"), on="key", how="left"))
assert (d.num_faces <= d.image_count.fillna(np.inf)).all()      # at most one face per image
d["p_face"] = d.num_faces / d.image_count
d["single"] = ~d.key.str.contains(" ")

# ------------------------------------------------------------------ helpers ---------------------------------------
def bin_means(x, y, k=N_BINS):
    g = pd.DataFrame({"x": x, "y": y}); g["bin"] = pd.qcut(g.x.rank(method="first"), k, labels=False)
    return g.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size")).reset_index()

def loess_operator(x, x0s, span=0.75, degree=2):
    n = len(x); q = int(np.floor(n * span + 1e-5)); L = np.zeros((len(x0s), n))
    for j, x0 in enumerate(x0s):
        dist = np.abs(x - x0); h = np.sort(dist)[q - 1]; w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
        X = np.vander(x - x0, degree + 1, increasing=True); XtW = X.T * w
        L[j] = (np.linalg.pinv(XtW @ X) @ XtW)[0]
    return L

def loess_band(x, y, span=0.75, degree=2, n_grid=80):
    x = np.asarray(x, float); y = np.asarray(y, float); xg = np.linspace(x.min(), x.max(), n_grid)
    Lg, Ld = loess_operator(x, xg, span, degree), loess_operator(x, x, span, degree)
    resid = y - Ld @ y; M = np.eye(len(x)) - Ld; MtM = M.T @ M; d1, d2 = np.trace(MtM), np.trace(MtM @ MtM)
    se = np.sqrt((resid ** 2).sum() / d1 * (Lg ** 2).sum(axis=1)); tq = stats.t.ppf(0.975, d1 ** 2 / d2); f = Lg @ y
    return xg, f, f - tq * se, f + tq * se

def corr(x, y):
    r, p = stats.pearsonr(x, y); n = len(x); z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return dict(n=int(n), r=float(r), lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)), p=float(p))

def ylim_for(ys, parity, top=0.14):
    lo, hi = min(map(np.min, ys)), max(map(np.max, ys))
    if parity: lo, hi = min(lo, 0.5), max(hi, 0.5)
    r = hi - lo; lo, hi = lo - 0.08 * r, hi + top * r; r = hi - lo      # figure_03.R: coord_cartesian(ylim = range + c(-0.08, 0.14) * diff)
    return lo - 0.05 * r, hi + 0.05 * r                                 # ... plus ggplot's 5% expansion

def y_step(r): return 0.05 if r > 0.2 else 0.02 if r > 0.07 else 0.01

def draw_panel(ax, b, xlab, ylab, inset_lines, ylim, parity):
    xg, f, lo, hi = loess_band(b.x, b.y)
    if parity: ax.axhline(0.5, color="black", lw=1.0 * SCALE, zorder=1)
    ax.fill_between(xg, lo, hi, color=BAND_COL, alpha=0.4, lw=0, zorder=2)
    ax.plot(xg, f, color=LOESS_COL, lw=3.0 * SCALE, zorder=3)
    ax.plot(b.x, b.y, "o", color="black", ms=15.5 * SCALE, zorder=4)
    for sp in ax.spines.values(): sp.set_color("black"); sp.set_linewidth(1.2 * SCALE)
    ax.tick_params(labelsize=FS["tick"], colors="black", labelcolor="#333333", length=5 * SCALE)
    ax.set_xlabel(xlab, fontsize=FS["title"], labelpad=10 * SCALE, linespacing=1.05)
    ax.set_ylabel(ylab, fontsize=FS["title"], labelpad=8 * SCALE, linespacing=1.05)
    xr = b.x.max() - b.x.min(); ax.set_xlim(b.x.min() - 0.05 * xr, b.x.max() + 0.05 * xr)
    ax.set_ylim(*ylim); ax.yaxis.set_major_locator(MultipleLocator(y_step(ylim[1] - ylim[0])))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.text(0.975, 0.965, "\n".join(inset_lines), transform=ax.transAxes, ha="right", va="top", fontsize=FS["inset"], linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.25", fc="white", ec="black", lw=0.9 * SCALE), zorder=6)

def draw_quartile_panel(ax, g, xcol, ycol, xlab, ylab, inset_lines):
    q = pd.qcut(g[xcol].rank(method="first"), 4, labels=False)
    t = g.groupby(q).agg(y=(ycol, "mean"), sd=(ycol, "std"), n=(ycol, "size"), lo_x=(xcol, "min"), hi_x=(xcol, "max")).reset_index(drop=True)
    t["ci"] = 1.96 * t.sd / np.sqrt(t.n)
    ax.bar(t.index, t.y, width=0.72, color="#B3B3B3", edgecolor="black", linewidth=1.0 * SCALE, zorder=2)
    ax.errorbar(t.index, t.y, yerr=t.ci, fmt="none", ecolor="black", elinewidth=1.6 * SCALE, capsize=6 * SCALE, capthick=1.6 * SCALE, zorder=3)
    for sp in ax.spines.values(): sp.set_color("black"); sp.set_linewidth(1.2 * SCALE)
    ax.tick_params(labelsize=FS["tick"], colors="black", labelcolor="#333333", length=5 * SCALE)
    ax.set_xticks(t.index); ax.set_xticklabels(["Q1\n(abstract)", "Q2", "Q3", "Q4\n(concrete)"], linespacing=1.05)
    ax.set_xlabel(xlab, fontsize=FS["title"], labelpad=10 * SCALE, linespacing=1.05)
    ax.set_ylabel(ylab, fontsize=FS["title"], labelpad=8 * SCALE, linespacing=1.05)
    ax.set_xlim(-0.6, 3.6)
    lo, top = PANEL_A_YLIM if PANEL_A_YLIM else (0, t.y.max() * 1.55); ax.set_ylim(lo, top)
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if top - lo > 0.3 else 0.02)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.text(0.975, 0.965, "\n".join(inset_lines), transform=ax.transAxes, ha="right", va="top", fontsize=FS["inset"], linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.25", fc="white", ec="black", lw=0.9 * SCALE), zorder=6)
    return t

def tag_panels(fig, axes_tags):
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for ax, tag in axes_tags:
        bb = ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
        fig.text(max(bb.x0 - 0.008, 0.002), bb.y1 + 0.006, tag, fontsize=FS["tag"], fontweight="bold", ha="left", va="bottom")

# ------------------------------------------------------------------ panels B and C (unchanged recipe) --------------
gB = d.dropna(subset=["conc"]); gC = d.dropna(subset=["perc"])
bB, bC = bin_means(gB.conc, gB.frac_male), bin_means(gC.perc, gC.frac_male)
ylim_BC = ylim_for([bB.y, bC.y], parity=True)
stats_out = {"B": corr(gB.conc, gB.frac_male) | dict(bin_first_last=[float(bB.y.iloc[0]), float(bB.y.iloc[-1])]),
             "C": corr(gC.perc, gC.frac_male) | dict(bin_first_last=[float(bC.y.iloc[0]), float(bC.y.iloc[-1])])}
bins_out = [bB.assign(panel="B"), bC.assign(panel="C")]

# ------------------------------------------------------------------ panel A, both vocabularies ---------------------
for vocab in ["all", "single"]:
    gA = d.dropna(subset=["conc", "image_count"]); gA = gA if vocab == "all" else gA[gA.single]
    bA = bin_means(gA.conc, gA.p_face)
    n_img = int(gA.image_count.sum())
    stats_out[f"A_{vocab}"] = corr(gA.conc, gA.p_face) | dict(bin_first_last=[float(bA.y.iloc[0]), float(bA.y.iloc[-1])],
                                                           n_images=n_img, n_faces=int(gA.num_faces.sum()),
                                                           pooled_p_face=float(gA.num_faces.sum() / n_img))
    bins_out.append(bA.assign(panel=f"A_{vocab}"))
    fig, axes = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H))
    inset_A = [N_IMAGES_LABEL, f"N = {len(gA):,} search terms"]   # same image-set label as panels B and C
    if PANEL_A_STYLE == "quartiles":
        qA = draw_quartile_panel(axes[0], gA, "conc", "p_face", "Concreteness Quartile\n(Human Judgments)", "P(Human Face in Google Image)", inset_A)
        stats_out[f"A_{vocab}"]["quartiles"] = [dict(q=i + 1, mean=float(r.y), ci=float(r.ci), n=int(r.n), conc_range=[float(r.lo_x), float(r.hi_x)]) for i, r in qA.iterrows()]
        bins_out.append(qA.rename(columns={"lo_x": "x_min", "hi_x": "x_max"}).assign(panel=f"A_{vocab}_quartiles"))
    else:
        draw_panel(axes[0], bA, "Average Concreteness\n(Human Judgments)", "P(Human Face in Google Image)",
                   inset_A, ylim_for([bA.y], parity=False, top=0.55), parity=False)   # head-room: the inset sits above the highest bins
    draw_panel(axes[1], bB, "Average Concreteness\n(Human Judgments)", "P(Male Face in Google Image)",
               [N_IMAGES_LABEL, f"N = {len(gB):,} search terms"], ylim_BC, parity=True)
    draw_panel(axes[2], bC, "Average Perceptual Strength\n(Human Judgments)", "P(Male Face in Google Image)",
               [N_IMAGES_LABEL, f"N = {len(gC):,} search terms"], ylim_BC, parity=True)
    fig.tight_layout(w_pad=3.0, rect=(0.005, 0, 1, 1 - 0.07 * SCALE))
    tag_panels(fig, [(axes[0], "A"), (axes[1], "B"), (axes[2], "C")])
    stem = FIGURE_STEM + ("_quartiles" if PANEL_A_STYLE == "quartiles" else "") + ("" if vocab == "all" else "_singlewords")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), dpi=DPI)
    plt.close(fig)

pd.concat(bins_out).to_csv(os.path.join(OUT_DIR, f"{FIGURE_STEM}_bins.csv"), index=False)
json.dump(stats_out, open(os.path.join(OUT_DIR, f"{FIGURE_STEM}_stats.json"), "w"), indent=1)
for k, s in stats_out.items():
    print(f"{k:9s} r = {s['r']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] P = {s['p']:.1e}; bins {s['bin_first_last'][0]:.3f} -> {s['bin_first_last'][1]:.3f}; n = {s['n']:,}"
          + (f"; images = {s['n_images']:,}; pooled P(face) = {s['pooled_p_face']:.3f}" if "n_images" in s else ""))
    for qd in s.get("quartiles", []):
        print(f"           Q{qd['q']} (conc {qd['conc_range'][0]:.2f}-{qd['conc_range'][1]:.2f}, n = {qd['n']:,}): P(face) = {qd['mean']:.4f} +/- {qd['ci']:.4f}")
print("written to", OUT_DIR)
