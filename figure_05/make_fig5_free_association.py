#!/usr/bin/env python3
"""
Figure 5 (free-association data), single panel.

Pearson r (cue concreteness x response gender) by gender scorer, one line per
dataset/generator. Filled markers = significant after Bonferroni correction
(60 tests, alpha = 8.3e-04); hollow = not. Error bars are 95% CIs (Fisher z).
Band text:  red region (r > 0)  "Concrete -> Male"
            blue region (r < 0) "Abstract -> Male"

15 Oct 2026: the heat-map panel (former panel b) was dropped as redundant --
it showed exactly the 60 correlations plotted here. No panel letter is drawn.

Correlations are the 60 values of the 6 Oct 2026 registry run (3 d.p.);
95% CIs and two-sided p-values are recomputed from r and n. Set CORR_CSV to
results/figure_05/master12_correlations.csv for full precision via GC_CORR_CSV.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as t_dist

# ------------------------------------------------------------------ CONFIG
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import results
OUT = results("figure_05")
STEM = "fig5_free_association"
CORR_CSV = os.environ.get("GC_CORR_CSV")  # optional full-precision compute output

SCORERS = ["word2vec", "MPNet", "GPT4", "FastText", "Qwen3", "distilBERT",
           "Nomic v2 MoE", "Wikipedia GloVe", "RoBERTa", "BERT", "GPT2", "ConceptNet"]

ROWS = [("SWOW", "Humans"), ("LWOW", "Humans"), ("LWOW", "Llama3"),
        ("LWOW", "Haiku"), ("LWOW", "Mistral")]

R = np.array([
    [-0.311, -0.310, -0.205, -0.228, -0.198, -0.253, -0.223, -0.250, -0.182, -0.038, +0.005, -0.004],
    [-0.305, -0.309, -0.205, -0.221, -0.193, -0.253, -0.218, -0.243, -0.193, -0.040, +0.008, +0.002],
    [-0.307, -0.259, -0.347, -0.283, -0.309, -0.211, -0.289, -0.198, -0.201, -0.210, -0.115, +0.102],
    [-0.277, -0.230, -0.284, -0.227, -0.237, -0.230, -0.207, -0.206, -0.216, -0.037, -0.077, +0.045],
    [-0.256, -0.186, -0.221, -0.201, -0.181, -0.165, -0.173, -0.191, -0.204, +0.024, -0.024, +0.048],
])
# cue counts as reported in results.tex (6 Oct 2026 registry run)
N_CUES = {("SWOW", "Humans"): 11179, ("LWOW", "Humans"): 10495, ("LWOW", "Llama3"): 10496,
          ("LWOW", "Haiku"): 10527, ("LWOW", "Mistral"): 10527}

COLORS = {"Humans": "#F8766D", "Llama3": "#00BA38", "Haiku": "#00B9E3", "Mistral": "#C77CFF"}
RED, BLUE = "#B2182B", "#2166AC"
BAND_RED, BAND_BLUE, GRID = "#FBEFEF", "#EEF3FA", "#E6E6E6"

ALPHA = 0.05 / (R.shape[0] * R.shape[1])          # Bonferroni, 60 tests


# ------------------------------------------------------------------ data
def load():
    recs = []
    if CORR_CSV:
        d = pd.read_csv(CORR_CSV)
        for (ds, gen) in ROWS:
            for sc in SCORERS:
                row = d[(d.dataset == ds) & (d.generator == gen) & (d.scorer == sc)].iloc[0]
                recs.append((ds, gen, sc, float(row.r),
                             int(row["n_cues"]) if "n_cues" in d.columns else
                             int(row["n"]) if "n" in d.columns else N_CUES[(ds, gen)]))
    else:
        for i, (ds, gen) in enumerate(ROWS):
            for j, sc in enumerate(SCORERS):
                recs.append((ds, gen, sc, R[i, j], N_CUES[(ds, gen)]))
    d = pd.DataFrame(recs, columns=["dataset", "generator", "scorer", "r", "n"])

    z, se = np.arctanh(d.r), 1 / np.sqrt(d.n - 3)
    d["ci_lo"], d["ci_hi"] = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    tval = d.r * np.sqrt((d.n - 2) / (1 - d.r ** 2))
    d["p"] = 2 * t_dist.sf(np.abs(tval), d.n - 2)
    d["sig"] = d.p < ALPHA
    return d


# ------------------------------------------------------------------ panel
def draw(ax, d):
    ax.set_xlim(-0.5, len(SCORERS) - 0.5)
    ax.set_ylim(-0.429, 0.184)

    ax.axhspan(0, 1, color=BAND_RED, zorder=0, lw=0)
    ax.axhspan(-1, 0, color=BAND_BLUE, zorder=0, lw=0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=1.1, zorder=2)

    ax.text(0.55, 0.0215, "Concrete \u2192 Male", transform=ax.get_yaxis_transform(),
            ha="center", va="center", fontsize=17.5, fontweight="bold", color=RED, zorder=6)
    ax.text(0.55, -0.0215, "Abstract \u2192 Male", transform=ax.get_yaxis_transform(),
            ha="center", va="center", fontsize=17.5, fontweight="bold", color=BLUE, zorder=6)

    handles = []
    x = np.arange(len(SCORERS))
    for ds, gen in ROWS:
        sub = (d[(d.dataset == ds) & (d.generator == gen)]
               .set_index("scorer").loc[SCORERS])
        col = COLORS[gen]
        ls = "-" if ds == "SWOW" else "--"
        mk = "o" if ds == "SWOW" else "s"

        ax.plot(x, sub.r, ls, color=col, linewidth=2.5, zorder=3)
        ax.errorbar(x, sub.r, yerr=[sub.r - sub.ci_lo, sub.ci_hi - sub.r],
                    fmt="none", ecolor=col, elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
        sig = sub.sig.values
        ax.plot(x[sig], sub.r.values[sig], mk, color=col, markersize=12,
                markeredgecolor="white", markeredgewidth=1.0, linestyle="none", zorder=5)
        ax.plot(x[~sig], sub.r.values[~sig], mk, markerfacecolor="white", markeredgecolor=col,
                markersize=12, markeredgewidth=2.6, linestyle="none", zorder=5)

        handles.append(Line2D([], [], color=col, linestyle=ls, linewidth=2.5, marker=mk,
                              markersize=11, markeredgecolor="white", markeredgewidth=1.0,
                              label=f"{ds} \u2014 {gen}"))

    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=13.5, frameon=True,
                     framealpha=0.94, edgecolor="#CCCCCC", handlelength=2.6, borderpad=0.6,
                     labelspacing=0.45)
    leg1.set_zorder(7)
    ax.add_artist(leg1)

    sig_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=11, color="#444444",
               label=f"Significant (Bonferroni, $p$ < {ALPHA:.1e})"),
        Line2D([], [], marker="o", linestyle="none", markersize=11, markerfacecolor="white",
               markeredgecolor="#444444", markeredgewidth=2.2, label="Not Significant"),
    ]
    leg2 = ax.legend(handles=sig_handles, title="Marker Fill", loc="lower right",
                     fontsize=13.5, title_fontsize=13.5, frameon=True, framealpha=0.94,
                     edgecolor="#CCCCCC", handlelength=1.4, borderpad=0.7, labelspacing=0.45)
    leg2.set_zorder(7)

    ax.set_xticks(x)
    ax.set_xticklabels(SCORERS, rotation=30, ha="right", rotation_mode="anchor", fontsize=15)
    ax.set_yticks(np.arange(-0.4, 0.15, 0.1))
    ax.tick_params(axis="y", labelsize=14, length=5)
    ax.tick_params(axis="x", length=5)
    ax.set_xlabel("Embedding Used to Score Response Gender  (Ordered by Mean $r$)",
                  fontsize=17, labelpad=14)
    ax.set_ylabel("Pearson $r$ (Cue Concreteness \u00d7 Response Gender)",
                  fontsize=16.5, labelpad=12)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_color("black")


# ------------------------------------------------------------------ figure
def main():
    mpl.rcParams["font.family"] = "DejaVu Sans"
    d = load()

    W, H = 4500, 2450                        # same axes geometry as the two-panel version (300 dpi)
    fig = plt.figure(figsize=(W / 300, H / 300))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([744 / W, 1 - 1942 / H, (3977 - 744) / W, (1942 - 108) / H])
    draw(ax, d)

    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(os.path.join(OUT, f"{STEM}.{ext}"), dpi=dpi, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
    plt.close(fig)

    print(f"Bonferroni alpha = {ALPHA:.2e}")
    print("non-significant cells:")
    print(d[~d.sig][["dataset", "generator", "scorer", "r", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
