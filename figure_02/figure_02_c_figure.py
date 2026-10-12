"""figure_02_c_figure.py  (= fig2_by_domain/03_figure.py, 5 Oct 2026)

Fig. 2 (fig2_by_domain): A/B within-domain abstract-third vs concrete-third contrast for words and phrases; C/D construal-class
centroids per model with the equal-weight mean across models. Reads results/figure_02/fig2_terciles_*.csv and fig2_centroids_*.csv;
writes results/figure_02/fig2_by_domain.png / .pdf."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from paths import results
from registry import REGISTRY, SHAPES
OUT = results("figure_02")
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

tw = pd.read_csv(f"{OUT}/fig2_terciles_words.csv"); tp = pd.read_csv(f"{OUT}/fig2_terciles_phrases.csv")
cw = pd.read_csv(f"{OUT}/fig2_centroids_words.csv"); cp = pd.read_csv(f"{OUT}/fig2_centroids_phrases.csv")

def cls_group(c):
    c = str(c)
    if c.startswith("Verb"): return "verb"
    if c in ("Adverb", "Modifier phrase", "Adjective"): return "modifier"
    if "entity" in c: return "entity noun"
    if "event" in c: return "event/state noun"
    return "other"
GCOL = {"verb": "#c0392b", "modifier": "#e67e22", "event/state noun": "#5dade2", "entity noun": "#1f4e79", "other": "#95a5a6"}
COL = {"Verb": "#c0392b", "Verb phrase / clause": "#c0392b", "Adverb": "#e67e22", "Modifier phrase": "#e67e22",
       "Adjective": "#f1c40f", "Noun: event/state": "#5dade2", "Noun phrase: event/state": "#5dade2",
       "Noun: entity": "#1f4e79", "Noun phrase: entity": "#1f4e79"}
CLS_LABEL = {"Verb phrase / clause": "Verb phrase", "Modifier phrase": "Adverb phrase"}
RENAME = {"noun.Tops": "noun.GenericCategory"}
MODELS = [n for n, _ in REGISTRY]
BIG = ("*", "P", "X", "x", "+", "1")

def tercile_panel(ax, t, title):
    t = t.sort_values("diff_mean", ascending=True).reset_index(drop=True)      # largest gap at the top
    CON, ABS = "#9a9a9a", "black"
    for j, r in t.iterrows():
        ax.errorbar(r.concrete_mean, j - 0.14, xerr=r.concrete_ci, fmt="o", color=CON, ecolor=CON, ms=9, capsize=3, lw=1.3, zorder=2)
        ax.errorbar(r.abstract_mean, j + 0.14, xerr=r.abstract_ci, fmt="^", color=ABS, ecolor=ABS, ms=10, capsize=3, lw=1.3, zorder=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{RENAME.get(d, d)}  (N={n:,})" for d, n in zip(t.domain, t.N_terms)], fontsize=11.5)
    for lab_, c_ in zip(ax.get_yticklabels(), t.cls):
        lab_.set_color(GCOL[cls_group(c_)]); lab_.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.axvline(0, color="k", lw=.5); ax.set_ylim(-0.8, len(t) - 0.2)
    lim = 1.08 * max(abs(t.concrete_mean - t.concrete_ci).max(), abs(t.concrete_mean + t.concrete_ci).max(),
                     abs(t.abstract_mean - t.abstract_ci).max(), abs(t.abstract_mean + t.abstract_ci).max())
    ax.set_xlim(-lim, lim * 1.18)
    for j, r in t.iterrows():                                               # x marks the domains whose abstract third is more male
        if r.diff_mean > 0: ax.text(lim * 1.12, j, "x", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=22)
    ax.text(0.5, 1.012, "x indicates abstract is more male than concrete", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color="#333")
    ax.set_xlabel(""); ax.set_ylabel("")
    ylab = -0.045
    ax.text(0.50, ylab, "Gender Association (z-score)", transform=ax.transAxes, ha="center", va="top", fontsize=14)
    ax.text(0.12, ylab, "\u2190 More Female", transform=ax.transAxes, ha="center", va="top", fontsize=13, fontstyle="italic")
    ax.text(0.88, ylab, "More Male \u2192", transform=ax.transAxes, ha="center", va="top", fontsize=13, fontstyle="italic")

def centroid_panel(ax, c, title):
    for _, r in c[c.model != "mean_across_models"].iterrows():
        mk = SHAPES[r.model]; filled = mk not in ("x", "+", "1")
        ax.plot(r.z_conc, r.z_gender, mk, color=COL[r.cls], ms=12 if mk in BIG else 9, alpha=.9,
                mec="white" if filled else COL[r.cls], mew=.6 if filled else 2.0, zorder=2)
    for _, r in c[c.model == "mean_across_models"].iterrows():
        ax.errorbar(r.z_conc, r.z_gender, xerr=r.ci_conc, yerr=r.ci_gender, fmt="o", color=COL[r.cls], ms=19, mec="black", mew=1.8,
                    ecolor="black", elinewidth=1.4, capsize=4, zorder=3)
        ax.annotate(CLS_LABEL.get(r.cls, r.cls), (r.z_conc, r.z_gender), fontsize=13, fontweight="bold", xytext=(16, 0), textcoords="offset points",
                    va="center", color=COL[r.cls], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.85), zorder=4)
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    xl = 1.12 * c.z_conc.abs().max() + 0.45; yl = 1.12 * c.z_gender.abs().max()
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    ylab = -0.07
    ax.text(0.50, ylab, "Mean Concreteness (z-score)", transform=ax.transAxes, ha="center", va="top", fontsize=14)
    ax.text(0.13, ylab, "\u2190 More Abstract", transform=ax.transAxes, ha="center", va="top", fontsize=13, fontstyle="italic")
    ax.text(0.87, ylab, "More Concrete \u2192", transform=ax.transAxes, ha="center", va="top", fontsize=13, fontstyle="italic")
    xlab = -0.085
    ax.text(xlab, 0.50, "Gender Association (z-score)", transform=ax.transAxes, ha="right", va="center", fontsize=14, rotation=90)
    ax.text(xlab, 0.13, "\u2190 More Female", transform=ax.transAxes, ha="right", va="center", fontsize=13, fontstyle="italic", rotation=90)
    ax.text(xlab, 0.87, "More Male \u2192", transform=ax.transAxes, ha="right", va="center", fontsize=13, fontstyle="italic", rotation=90)

fig = plt.figure(figsize=(23, 25))
gs_top = fig.add_gridspec(1, 2, left=0.16, right=0.985, top=0.965, bottom=0.46, wspace=0.50)
gs_bot = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.385, bottom=0.09, wspace=0.30)
axA, axB = fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])
axC, axD = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])
tercile_panel(axA, tw, "Words"); tercile_panel(axB, tp, "Phrases")
centroid_panel(axC, cw, "Words"); centroid_panel(axD, cp, "Phrases")

h_terc = [Line2D([0], [0], marker="o", color="#9a9a9a", lw=0, ms=11, label="most concrete third (mean +/- 95% CI across models)"),
          Line2D([0], [0], marker="^", color="black", lw=0, ms=12, label="most abstract third (mean +/- 95% CI across models)")]
fig.legend(handles=h_terc, loc="center", bbox_to_anchor=(0.57, 0.418), ncol=2, fontsize=13, frameon=False)
h_mod = [Line2D([0], [0], marker=SHAPES[m], color="#555", lw=0, ms=11 if SHAPES[m] in BIG else 9, mec="#555", label=m) for m in MODELS]
h_mod.append(Line2D([0], [0], marker="o", color="#bbb", lw=0, ms=14, mec="black", mew=1.6, label="mean across models (+/- 95% CI)"))
fig.legend(handles=h_mod, loc="center", bbox_to_anchor=(0.53, 0.03), ncol=5, fontsize=12.5, frameon=False,
           title="model (marker shape); colour = construal class", title_fontsize=12.5)

fig.canvas.draw(); rend = fig.canvas.get_renderer()
for ax, letter in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
    bb = ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
    fig.text(max(bb.x0 - 0.008, 0.004), bb.y1 + 0.003, letter, fontsize=24, fontweight="bold", ha="left", va="bottom")
plt.savefig(f"{OUT}/fig2_by_domain.png", dpi=170); plt.savefig(f"{OUT}/fig2_by_domain.pdf")
print("saved", f"{OUT}/fig2_by_domain.png")
