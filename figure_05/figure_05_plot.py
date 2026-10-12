#!/usr/bin/env python3
"""figure_05_plot.py

Fig. 5 (fig5_free_association), stacked:
  A  Pearson r (cue concreteness x response gender) by gender scorer, one line per
     dataset/generator, 95% CIs, markers hollow when not significant after Bonferroni
  B  the same correlations as a heat map

Reads results/figure_05/master12_correlations.csv (written by figure_05_compute.py). Scorers are ordered by
mean r, as the x-axis label states; significance is Bonferroni over all N_TESTS = 60 cells.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import results

# ================================ CONFIG ======================================
OUT       = results("figure_05") + "/"
OUT_STUB  = "fig5_free_association"
TABLE_CSV = os.path.join(OUT, "master12_correlations.csv")       # written by figure_05_compute.py

SERIES  = ["SWOW \u2014 Humans", "LWOW \u2014 Humans", "LWOW \u2014 Llama3", "LWOW \u2014 Haiku", "LWOW \u2014 Mistral"]
N_TESTS = 60                                                      # 5 sources x 12 scorers (Bonferroni family)

COLORS  = ["#F8766D", "#F8766D", "#00BA38", "#00B9E3", "#C77CFF"]
MARKERS = ["o", "s", "s", "s", "s"]
LSTYLES = ["-", "--", "--", "--", "--"]

FS = dict(tick=14, scorer=14.5, label=16.5, legend=13.5, annot=16, cell=13.5, ns=11.5, cbar=13.5, tag=26)
Y_LAB_A = "Pearson $r$ (Cue Concreteness \u00d7 Response Gender)"
X_LAB_A = "Embedding Used to Score Response Gender  (Ordered by Mean $r$)"
FIG_W, FIG_H = 15.0, 14.6
# ============================== END CONFIG ====================================


SCORERS = []   # filled from the table, ordered by mean r


def build_table():
    global SCORERS
    if not os.path.exists(TABLE_CSV):
        raise SystemExit(f"{TABLE_CSV} not found -- run figure_05_compute.py first")
    t = pd.read_csv(TABLE_CSV)
    t["series"] = t["dataset"] + " \u2014 " + t["generator"]
    t = t.rename(columns={"ci_low": "ci_lo", "ci_high": "ci_hi"})
    t = t[t["series"].isin(SERIES)]
    t["significant"] = t["p"] < 0.05 / N_TESTS
    SCORERS = list(t.groupby("scorer")["r"].mean().sort_values().index)   # x-axis order: by mean r
    t["s"] = t["series"].map({v: i for i, v in enumerate(SERIES)})
    t["j"] = t["scorer"].map({v: i for i, v in enumerate(SCORERS)})
    return t.dropna(subset=["s", "j"])


def panel_a(ax, t):
    ax.set_facecolor("white")
    ax.axhspan(0, 1, color="#FBEFEF", zorder=0)
    ax.axhspan(-1, 0, color="#EEF3FA", zorder=0)
    ax.grid(True, color="#E6E6E6", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=1.3, zorder=2)
    offsets = np.linspace(-0.16, 0.16, len(SERIES))
    for s in range(len(SERIES)):
        d = t[t.s == s].sort_values("j")
        x = d.j.values + offsets[s]
        col = COLORS[s]
        ax.plot(x, d.r, LSTYLES[s], color=col, linewidth=2.4, zorder=3)
        ax.errorbar(x, d.r, yerr=[d.r - d.ci_lo, d.ci_hi - d.r], fmt="none", ecolor=col,
                    elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)
        sig = d.significant.values.astype(bool)
        ax.scatter(x[sig], d.r.values[sig], marker=MARKERS[s], s=110, facecolor=col,
                   edgecolor="white", linewidth=0.9, zorder=5)
        ax.scatter(x[~sig], d.r.values[~sig], marker=MARKERS[s], s=125, facecolor="white",
                   edgecolor=col, linewidth=2.2, zorder=5)
    ax.set_xlim(-0.5, len(SCORERS) - 0.5)
    ax.set_ylim(-0.425, 0.182)
    ax.set_xticks(range(len(SCORERS)))
    ax.set_xticklabels(SCORERS, rotation=28, ha="right", rotation_mode="anchor", fontsize=FS["scorer"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.tick_params(length=5)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\u2212")))
    ax.set_ylabel(Y_LAB_A, fontsize=FS["label"], labelpad=10)
    ax.set_xlabel(X_LAB_A, fontsize=FS["label"], labelpad=12)
    ax.text(0.55, 0.021, "Concrete \u2192 Male", transform=ax.get_yaxis_transform(), ha="center",
            va="center", fontsize=FS["annot"], fontweight="bold", color="#B2182B")
    ax.text(0.55, -0.0195, "Concrete \u2192 Female", transform=ax.get_yaxis_transform(), ha="center",
            va="center", fontsize=FS["annot"], fontweight="bold", color="#2166AC")
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)

    h1 = [Line2D([], [], color=COLORS[s], linestyle=LSTYLES[s], linewidth=2.4, marker=MARKERS[s],
                 markersize=10, markeredgecolor="white", label=SERIES[s]) for s in range(len(SERIES))]
    leg1 = ax.legend(handles=h1, loc="upper left", fontsize=FS["legend"], frameon=True, framealpha=0.95,
                     edgecolor="#CCCCCC", handlelength=2.6)
    ax.add_artist(leg1)
    h2 = [Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="#444444",
                 markeredgecolor="#444444", label=f"Significant (Bonferroni, $p$ < {0.05 / N_TESTS:.1e})"),
          Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="white",
                 markeredgecolor="#444444", markeredgewidth=2.0, label="Not Significant")]
    ax.legend(handles=h2, title="Marker Fill", loc="lower right", fontsize=FS["legend"] - 0.5,
              title_fontsize=FS["legend"], frameon=True, framealpha=0.95, edgecolor="#CCCCCC")


def panel_b(ax, cax, t):
    M = t.pivot(index="s", columns="j", values="r").values
    S = t.pivot(index="s", columns="j", values="significant").values.astype(bool)
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35)
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="both", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rgb = cmap(norm(M[i, j]))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            tc = "white" if lum < 0.45 else "#1A1A1A"
            ax.text(j, i - (0.10 if not S[i, j] else 0.0) - 0.02, f"{M[i, j]:+.3f}".replace("+0.000", "0.000"),
                    ha="center", va="center", fontsize=FS["cell"], fontweight="bold", color=tc)
            if not S[i, j]:
                ax.text(j, i + 0.27, "n.s.", ha="center", va="center", fontsize=FS["ns"], style="italic", color=tc)
    ax.set_xticks(range(len(SCORERS)))
    ax.set_xticklabels(SCORERS, rotation=28, ha="right", rotation_mode="anchor", fontsize=FS["scorer"])
    ax.set_yticks(range(len(SERIES)))
    ax.set_yticklabels([s.replace(" \u2014 ", "  \u2014  ") for s in SERIES], fontsize=FS["scorer"] + 0.5)
    ax.tick_params(axis="x", pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Pearson $r$", fontsize=FS["label"], labelpad=10)
    cb.ax.tick_params(labelsize=FS["cbar"])
    cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\u2212")))
    cb.outline.set_visible(False)
    cb.ax.text(0.5, 1.045, "Abstract \u2192 Female", transform=cb.ax.transAxes, ha="center", va="bottom",
               fontsize=FS["cbar"] - 1, color="#B2182B")
    cb.ax.text(0.5, -0.17, "Abstract \u2192 Male", transform=cb.ax.transAxes, ha="center", va="top",
               fontsize=FS["cbar"] - 1, color="#2166AC")


def main():
    mpl.rcParams["font.family"] = "DejaVu Sans"
    t = build_table()
    print("source:", TABLE_CSV)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.55, 1.0], width_ratios=[1, 0.017],
                  hspace=0.50, wspace=0.025, left=0.165, right=0.905, top=0.975, bottom=0.115)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[1, 0]); cax = fig.add_subplot(gs[1, 1])
    panel_a(ax_a, t)
    panel_b(ax_b, cax, t)
    for ax, tag in ((ax_a, "A"), (ax_b, "B")):
        bb = ax.get_position()
        fig.text(0.012, bb.y1 + 0.004, tag, fontsize=FS["tag"], fontweight="bold", va="center", ha="left")
    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(f"{OUT}{OUT_STUB}.{ext}", dpi=dpi, facecolor="white")
    t.drop(columns=["s", "j"]).to_csv(f"{OUT}{OUT_STUB}_values.csv", index=False)
    print("saved", f"{OUT}{OUT_STUB}.png")


if __name__ == "__main__":
    main()
