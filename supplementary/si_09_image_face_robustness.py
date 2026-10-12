#!/usr/bin/env python3
# ==============================================================================
# si_09_image_face_robustness.py   (9 Oct 2026)
#
# SUPPLEMENTARY SECTION "Robustness of the image-based results":
# robustness of Fig. 3 (Google-image faces) to the zero-face coding and to the
# number of faces per search term (the number of a term's images in which a face
# was detected: the face counts never exceed the image counts and crowd terms have
# few "faces", i.e. at most one face was classified per image).
#
# Fig. 3 plots P(Male Face in Google Image) by (A) human judgements of concreteness
# and (B) Lancaster perceptual strength, averaging frac_male over ALL 39,956 search
# terms; the image file codes frac_male = 0 for the 4,043 terms with no recorded face.
#
# Writes (results/si_09/):
#   tab_S_image_agg.tex      Supplementary Table (label tab:S-image-agg): the aggregate
#                            association under three specifications, as columns --
#                            (1) all terms as in Fig. 3; (2) terms with >= MIN_FACES
#                            faces; (3) the same terms, adjusted for log(1 + faces).
#                            Rows: r, beta, P, twentieth means, n.
#   si_09_stats.json, si_09_bin_means.csv
#   figS_image_terms_with_faces.{png,pdf}, figS_image_face_count_control.{png,pdf}
#                            (if MAKE_FIGURES; Fig.-3-style figures of specifications
#                            (2), (3) and of tertiles of the number of faces)
#
# Conventions follow si_08_image_regressions.py (unstandardised OLS slopes, HC1
# robust SEs) so that the rows that overlap with Supplementary Table tab:S-image
# reproduce it exactly. Twentieth means are computed within the sample of each
# specification (20 equal-count bins of the predictor), as a figure drawn as Fig. 3
# from that sample would show them.
#
# Inputs : data/images/image_class_w_conc_PROCESSED.csv
#          data/norms/lancaster_sensorimotor_norms_2019.csv
# Paths resolve relative to the repo root (env GC_IMAGE_FILE, GC_LANC_FILE,
# GC_RESULTS_DIR override). Requires pandas, numpy, scipy, statsmodels, matplotlib.
# ==============================================================================
import os, sys, json, warnings, logging
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ================================ CONFIG ======================================
HERE          = os.path.dirname(os.path.abspath(__file__))
REPO          = os.environ.get("GC_REPO_DIR", os.path.dirname(HERE))
IMAGE_FILE    = os.environ.get("GC_IMAGE_FILE",    os.path.join(REPO, "data/images/image_class_w_conc_PROCESSED.csv"))
LANC_FILE     = os.environ.get("GC_LANC_FILE",     os.path.join(REPO, "data/norms/lancaster_sensorimotor_norms_2019.csv"))
OUT_DIR       = os.environ.get("GC_RESULTS_DIR",   os.path.join(REPO, "results/si_09"))

MIN_FACES    = 1        # a term enters the face-based specifications when it has at least this many faces
N_BINS       = 20       # equal-count bins of the predictor (as in Fig. 3)
MAKE_FIGURES = True
PERCEPTUAL   = ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"]
PREDS        = ["conc", "perc"]
# ============================== END CONFIG ====================================
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ data ------------------------------------------
# keep_default_na=False: the search terms "none", "null", "nan" are words, not missing values
im = pd.read_csv(IMAGE_FILE, low_memory=False, keep_default_na=False,
                 na_values={"Dom_Pos": ["", "NA"], "frac_males": ["NA", ""], "frac_females": ["NA", ""], "avg_age": ["NA", ""]})
for c in ["num_faces", "num_males", "num_females", "frac_male", "conc", "freq", "gender_conf"]:
    im[c] = pd.to_numeric(im[c])
assert (im.num_males + im.num_females == im.num_faces).all()
im["key"] = im["query"].str.lower().str.strip()
lan = pd.read_csv(LANC_FILE, keep_default_na=False)
for c in PERCEPTUAL:
    lan[c] = pd.to_numeric(lan[c])
lan["perc"] = lan[PERCEPTUAL].mean(axis=1)              # mean of the six perceptual modalities, as in Fig. 3b
lan["key"] = lan.Word.str.lower().str.strip()
d = im.merge(lan[["key", "perc"]].drop_duplicates("key"), on="key", how="left")
d["logf"] = np.log1p(d.num_faces)

meta = dict(n_terms=int(len(d)), n_zero_face=int((d.num_faces == 0).sum()), total_faces=int(d.num_faces.sum()),
            min_faces=MIN_FACES, n_bins=N_BINS)

# ------------------------------------------------------------------ helpers ---------------------------------------
def ntile(x, k): return pd.qcut(x.rank(method="first"), k, labels=False)

def bin_means(x, y, k=N_BINS):
    g = pd.DataFrame({"x": x, "y": y}); g["bin"] = ntile(g.x, k)
    return g.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size")).reset_index()

def corr(x, y):
    r, p = stats.pearsonr(x, y); n = len(x); z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return dict(n=int(n), r=float(r), lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)), p=float(p))

def partial_corr(x, y, Z):
    """Pearson correlation of x and y after both are residualised on the columns of Z; Fisher-z 95% CI."""
    Z = np.column_stack([np.ones(len(x)), Z]); k = Z.shape[1] - 1
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry); n = len(x); zf, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    return dict(n=int(n), r=float(r), lo=float(np.tanh(zf - 1.96 * se)), hi=float(np.tanh(zf + 1.96 * se)), p=float(p))

def fit(formula, data, term):
    """Unstandardised coefficient on `term` from OLS with HC1 robust SEs (as si_08)."""
    res = smf.ols(formula, data).fit(cov_type="HC1")
    ci = res.conf_int().loc[term]
    return dict(b=float(res.params[term]), se=float(res.bse[term]), stat=float(res.tvalues[term]), p=float(res.pvalues[term]),
                lo=float(ci[0]), hi=float(ci[1]), n=int(res.nobs))

def adjusted(data, controls):
    """frac_male net of its linear relation to the controls, re-centred on the grand mean (the quantity the figures plot)."""
    m = smf.ols("frac_male ~ " + " + ".join(controls), data).fit()
    return data.frac_male - m.fittedvalues + data.frac_male.mean()

# ------------------------------------------------------------------ aggregate specifications (tab:S-image-agg) ----
AGG = [  # key, sample, controls (None = bivariate)
    ("all",     lambda g: g,                              None),
    ("faces",   lambda g: g[g.num_faces >= MIN_FACES],    None),
    ("adj_f",   lambda g: g[g.num_faces >= MIN_FACES],    ["logf"]),
]
report, bins_out, binned = {"meta": meta}, [], {}
for pred in PREDS:
    base = d.dropna(subset=[pred]); r = {}
    for key, sample, controls in AGG:
        g = sample(base).copy()
        y = g.frac_male if controls is None else adjusted(g, controls)
        b = bin_means(g[pred], y); binned[(pred, key)] = b
        bins_out.append(b.assign(predictor=pred, specification=key))
        c = corr(g[pred], y) if controls is None else partial_corr(g[pred].values, g.frac_male.values, g[controls].values)
        s = fit(f"frac_male ~ {pred}" + ("" if controls is None else " + " + " + ".join(controls)), g, pred)
        r[key] = dict(r=c, slope=s, bin_first_last=[float(b.y.iloc[0]), float(b.y.iloc[-1])],
                      n_terms=int(len(g)), n_faces=int(g.num_faces.sum()), controls=controls or [])
    fb = base[base.num_faces >= MIN_FACES]
    r["corr_logf_x"] = float(np.corrcoef(fb.logf, fb[pred])[0, 1]); r["corr_logf_y"] = float(np.corrcoef(fb.logf, fb.frac_male)[0, 1])
    report[pred] = r

pd.concat(bins_out).to_csv(os.path.join(OUT_DIR, "si_09_bin_means.csv"), index=False)
json.dump(report, open(os.path.join(OUT_DIR, "si_09_stats.json"), "w"), indent=1)

# ------------------------------------------------------------------ LaTeX ------------------------------------------
def tex_num(x, dgt=3):
    t = f"{abs(x):.{dgt}f}"; return f"$-{t}$" if x < 0 else f"${t}$"
def tex_ci(lo, hi, dgt=3): return f"[{tex_num(lo, dgt)}, {tex_num(hi, dgt)}]"
def tex_p(p):
    if p < 2.2e-16: return r"$<2.2\times10^{-16}$"
    if p < 0.001:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        if round(m, 1) >= 10: m, e = m / 10, e + 1
        return rf"${m:.1f}\times10^{{{e}}}$"
    return f"${p:.3f}$"
def tex_n(n): return f"{n:,}".replace(",", "{,}")
EOL = r" \\"

# ---- Table 1: aggregate specifications as columns
AGG_HEAD = {"all":    r"\shortstack[c]{All terms\\(as in Fig.~3)$^{a}$}",
            "faces":  r"\shortstack[c]{Terms with\\$\geq$1 face}",
            "adj_f":  r"\shortstack[c]{Terms with $\geq$1 face,\\adjusted for faces$^{b}$}"}
BLOCKS = [("conc", r"\textit{Human judgements of concreteness (1--5; Fig.~3a)}", "most abstract twentieth", "most concrete twentieth"),
          ("perc", r"\textit{Human judgements of perceptual strength (0--5; Fig.~3b)}", "least perceptual twentieth", "most perceptual twentieth")]
keys = [k for k, _, _ in AGG]
def agg_rows(pred, lo_lab, hi_lab):
    S = report[pred]
    rows = [("Pearson's $r$",                           [tex_num(S[k]["r"]["r"]) for k in keys]),
            (r"\quad 95\% CI",                          [tex_ci(S[k]["r"]["lo"], S[k]["r"]["hi"]) for k in keys]),
            (r"OLS slope $\beta$ (per scale point)",    [tex_num(S[k]["slope"]["b"], 4) for k in keys]),
            (r"\quad 95\% CI (HC1)",                    [tex_ci(S[k]["slope"]["lo"], S[k]["slope"]["hi"], 4) for k in keys]),
            (r"\quad \textit{P}",                       [tex_p(S[k]["slope"]["p"]) for k in keys]),
            (f"Male fraction, {lo_lab}",                [f"{S[k]['bin_first_last'][0]:.3f}" for k in keys]),
            (f"Male fraction, {hi_lab}",                [f"{S[k]['bin_first_last'][1]:.3f}" for k in keys]),
            ("$n$ search terms",                        [tex_n(S[k]["n_terms"]) for k in keys])]
    return [f"{lab} & " + " & ".join(cells) + EOL for lab, cells in rows]
m = meta
L = [r"\begin{table}[!htbp]", r"  \centering",
     r"  \caption{\textbf{The association between conceptual abstraction and male faces in Google Images survives the",
     r"  exclusion of search terms without faces and adjustment for the number of faces.} Each column is a specification of",
     r"  the association between the fraction of male faces in a search term's images (0--1) and human judgements of its",
     r"  concreteness (upper block) or perceptual strength (lower block). $r$ is the two-tailed Pearson's correlation (in the",
     r"  adjusted specification, the partial correlation controlling for the number of faces); $\beta$ is the unstandardized",
     r"  ordinary least squares slope per scale point of the predictor, with heteroskedasticity-robust (HC1) 95\% confidence",
     r"  intervals and \textit{P} values; the twentieth means are the mean fraction of male faces in the lowest and highest of",
     r"  20 equal-count bins of the predictor, that is, the leftmost and rightmost points of a figure drawn as Fig.~3. $n$ is",
     r"  the number of search terms. " +
     rf"The terms with at least one face account for {tex_n(report['conc']['faces']['n_faces'])} faces ({tex_n(report['perc']['faces']['n_faces'])} among the terms with perceptual-strength ratings). " +
     rf"$^{{a}}$The {tex_n(m['n_zero_face'])} terms whose images contained no detectable face take the value zero, as in Fig.~3. " +
     r"$^{b}$Adjusted for the number of faces, that is, the number of a term's images in which a face was detected: $\beta$ is the",
     r"  coefficient on the predictor in a regression that includes $\log(1+\text{number of faces})$, and the twentieth means are of",
     r"  the fraction of male faces net of its linear relation to $\log(1+\text{number of faces})$, re-centred on the grand mean.}",
     r"  \label{tab:S-image-agg}",
     r"  \begingroup\footnotesize\setlength{\tabcolsep}{6pt}",
     r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule",
     "     & " + " & ".join(AGG_HEAD[k] for k in keys) + EOL, r"    \midrule"]
for pred, head, lo_lab, hi_lab in BLOCKS:
    L.append(rf"    \multicolumn{{4}}{{@{{}}l}}{{{head}}}" + EOL)
    L += ["    " + row for row in agg_rows(pred, lo_lab, hi_lab)]
    if pred == "conc": L.append(r"    \addlinespace[3pt]")
L += [r"    \bottomrule", r"  \end{tabular}\endgroup", r"\end{table}"]
open(os.path.join(OUT_DIR, "tab_S_image_agg.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ figures ----------------------------------------
if MAKE_FIGURES:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    Y_LAB = "P(Male Face in Google Image)"; Y_LAB_ADJ = "P(Male Face in Google Image)\nAdjusted for Number of Faces"
    X_LABS = {"conc": "Average Concreteness\n(Human Judgments)", "perc": "Average Perceptual Strength\n(Human Judgments)"}
    TERT_COLS = ["#E69F00", "#009E73", "#0072B2"]; LOESS_COL, BAND_COL = "#3366FF", "#999999"
    FS = dict(tick=20, title=22, tag=32, inset=17.5, legend=16); PANEL_W, PANEL_H, DPI = 7.0, 7.6, 400
    plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 18, "axes.linewidth": 1.2,
                         "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 5, "ytick.major.size": 5, "pdf.fonttype": 42})

    def loess_operator(x, x0s, span=0.75, degree=2):
        """Rows of the loess smoothing operator at x0s (R stats::loess defaults, direct fit)."""
        n = len(x); q = int(np.floor(n * span + 1e-5)); L = np.zeros((len(x0s), n))
        for j, x0 in enumerate(x0s):
            dist = np.abs(x - x0); h = np.sort(dist)[q - 1]; w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
            X = np.vander(x - x0, degree + 1, increasing=True); XtW = X.T * w
            L[j] = (np.linalg.pinv(XtW @ X) @ XtW)[0]
        return L
    def loess_band(x, y, span=0.75, degree=2, n_grid=80):
        """Fit and 95% limits on a grid, as ggplot2::geom_smooth(method = 'loess') draws them."""
        x = np.asarray(x, float); y = np.asarray(y, float); xg = np.linspace(x.min(), x.max(), n_grid)
        Lg, Ld = loess_operator(x, xg, span, degree), loess_operator(x, x, span, degree)
        resid = y - Ld @ y; M = np.eye(len(x)) - Ld; MtM = M.T @ M; d1, d2 = np.trace(MtM), np.trace(MtM @ MtM)
        se = np.sqrt((resid ** 2).sum() / d1 * (Lg ** 2).sum(axis=1)); tq = stats.t.ppf(0.975, d1 ** 2 / d2); f_ = Lg @ y
        return xg, f_, f_ - tq * se, f_ + tq * se
    def y_step(r): return 0.05 if r > 0.2 else 0.02 if r > 0.07 else 0.01
    def style_axes(ax, xlab, ylab):
        for sp in ax.spines.values(): sp.set_color("black"); sp.set_linewidth(1.2)
        ax.tick_params(labelsize=FS["tick"], colors="black", labelcolor="#333333", length=5)
        ax.set_xlabel(xlab, fontsize=FS["title"], labelpad=10, linespacing=1.05); ax.set_ylabel(ylab, fontsize=FS["title"], labelpad=8, linespacing=1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    def set_limits(ax, xs, ylim):
        xr = xs.max() - xs.min(); ax.set_xlim(xs.min() - 0.05 * xr, xs.max() + 0.05 * xr)
        ax.set_ylim(*ylim); ax.yaxis.set_major_locator(MultipleLocator(y_step(ylim[1] - ylim[0])))
    def shared_ylim(ys, top=0.14):
        lo, hi = min(map(np.min, ys)), max(map(np.max, ys)); r = hi - lo; lo, hi = lo - 0.08 * r, hi + top * r; r = hi - lo
        return lo - 0.05 * r, hi + 0.05 * r
    def inset(ax, lines):
        ax.text(0.975, 0.965, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=FS["inset"], linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.25", fc="white", ec="black", lw=0.9), zorder=6)
    def draw_curve(ax, b, colour=LOESS_COL, band=BAND_COL, band_alpha=0.4, ms=15.5, lw=3.0, dot_colour="black", zorder=3):
        xg, f_, lo, hi = loess_band(b.x, b.y)
        ax.fill_between(xg, lo, hi, color=band, alpha=band_alpha, lw=0, zorder=zorder - 1); ax.plot(xg, f_, color=colour, lw=lw, zorder=zorder)
        ax.plot(b.x, b.y, "o", color=dot_colour, ms=ms, zorder=zorder + 1)
    def tag_panels(fig, axes_tags):
        fig.canvas.draw(); rend = fig.canvas.get_renderer()
        for ax, tag in axes_tags:
            bb = ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
            fig.text(max(bb.x0 - 0.012, 0.003), bb.y1 + 0.006, tag, fontsize=FS["tag"], fontweight="bold", ha="left", va="bottom")
    def n_lines(pred, key): return [f"N = {report[pred][key]['n_faces']:,} faces", f"N = {report[pred][key]['n_terms']:,} search terms"]

    # figure 1: terms with faces (specification 3)
    ylim1 = shared_ylim([binned[(p, "faces")].y for p in PREDS])
    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_W, PANEL_H))
    for ax, pred in zip(axes, PREDS):
        b = binned[(pred, "faces")]; draw_curve(ax, b); style_axes(ax, X_LABS[pred], Y_LAB); set_limits(ax, b.x, ylim1); inset(ax, n_lines(pred, "faces"))
    fig.tight_layout(w_pad=3.0, rect=(0.01, 0, 1, 0.93)); tag_panels(fig, [(axes[0], "A"), (axes[1], "B")])
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT_DIR, f"figS_image_terms_with_faces.{ext}"), dpi=DPI)
    plt.close(fig)

    # figure 2: adjusted for faces (specification 4) and tertiles of the number of faces
    f = d[d.num_faces >= MIN_FACES]
    cuts = f.num_faces.quantile([1 / 3, 2 / 3]).round().astype(int).tolist()
    tert = pd.cut(f.num_faces, [0, cuts[0], cuts[1], f.num_faces.max()], labels=False)
    tert_lab = [f"{MIN_FACES}\u2013{cuts[0]} faces", f"{cuts[0] + 1}\u2013{cuts[1]} faces", f"{cuts[1] + 1}\u2013{int(f.num_faces.max())} faces"]
    strata = {pred: {t: bin_means(gg[pred], gg.frac_male) for t, gg in f.dropna(subset=[pred]).groupby(tert)} for pred in PREDS}
    report["tertiles"] = {pred: {tert_lab[t]: dict(corr(gg[pred], gg.frac_male), n=int(len(gg))) for t, gg in f.dropna(subset=[pred]).groupby(tert)} for pred in PREDS}
    json.dump(report, open(os.path.join(OUT_DIR, "si_09_stats.json"), "w"), indent=1)
    ylim_adj = shared_ylim([binned[(p, "adj_f")].y for p in PREDS]); ylim_str = shared_ylim([b.y for p in PREDS for b in strata[p].values()], top=0.45)
    fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_W, 2 * PANEL_H))
    for ax, pred in zip(axes[0], PREDS):
        b = binned[(pred, "adj_f")]; draw_curve(ax, b); style_axes(ax, X_LABS[pred], Y_LAB_ADJ); set_limits(ax, b.x, ylim_adj); inset(ax, n_lines(pred, "adj_f"))
    for ax, pred in zip(axes[1], PREDS):
        xs = pd.concat([b.x for b in strata[pred].values()])
        for t, b in strata[pred].items():
            draw_curve(ax, b, colour=TERT_COLS[t], band=TERT_COLS[t], band_alpha=0.18, ms=10.5, lw=2.6, dot_colour=TERT_COLS[t], zorder=3 + 2 * t)
        style_axes(ax, X_LABS[pred], Y_LAB); set_limits(ax, xs, ylim_str)
        handles = [Line2D([], [], color=TERT_COLS[t], marker="o", ms=10, lw=2.6, label=f"{tert_lab[t]}  (n = {report['tertiles'][pred][tert_lab[t]]['n']:,})") for t in range(3)]
        ax.legend(handles=handles, loc="upper right", fontsize=FS["legend"], frameon=True, edgecolor="black", fancybox=True, framealpha=1,
                  title="Faces per search term", title_fontsize=FS["legend"], handlelength=1.6, borderpad=0.6)
    fig.tight_layout(w_pad=3.0, h_pad=4.5, rect=(0.01, 0, 1, 0.965)); tag_panels(fig, [(axes[0, 0], "A"), (axes[0, 1], "B"), (axes[1, 0], "C"), (axes[1, 1], "D")])
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT_DIR, f"figS_image_face_count_control.{ext}"), dpi=DPI)
    plt.close(fig)

# ------------------------------------------------------------------ console report --------------------------------
def fp(p): return "<2.2e-16" if p < 2.2e-16 else f"{p:.1e}"
print(f"terms {m['n_terms']:,}; zero-face {m['n_zero_face']:,}; faces {m['total_faces']:,}")
for pred in PREDS:
    R = report[pred]; print(f"\n== {pred}")
    for key, _, _ in AGG:
        s_ = R[key]
        print(f"  {key:6s} r = {s_['r']['r']:+.3f} [{s_['r']['lo']:+.3f}, {s_['r']['hi']:+.3f}]; beta = {s_['slope']['b']:+.4f} [{s_['slope']['lo']:+.4f}, {s_['slope']['hi']:+.4f}] "
              f"P {fp(s_['slope']['p'])}; bins {s_['bin_first_last'][0]:.3f} -> {s_['bin_first_last'][1]:.3f}; n = {s_['n_terms']:,}")
    print(f"  among terms with faces: corr(log faces, x) = {R['corr_logf_x']:+.3f}; corr(log faces, frac_male) = {R['corr_logf_y']:+.3f}")
print("\nwritten to", OUT_DIR)
