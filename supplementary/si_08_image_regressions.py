#!/usr/bin/env python3
# ==============================================================================
# si_08_image_regressions.py  (= 13_manuscript_stats_fig3.py, 21 Sept 2026)
#
# SUPPLEMENTARY SECTION 8 / SUPPLEMENTARY TABLE 11  (and the Fig. 3 statistics in the main text)
#
# Reproduces every number reported for Fig. 3 (Google Images) in results.tex and
# in Supplementary Section "Robustness of the image-based results", and writes
# the body rows of that Supplementary Table:
#
#     fig3_image_stats.json        all statistics, machine-readable
#     tab_image_rows.tex           the rows inside the tabular of Supplementary Table 11
#
# Inputs:  image_class_w_conc_PROCESSED.csv   (one row per search term; frac_male
#          = fraction of male faces, 0 when no faces were detected)
#          lancaster_sensorimotor_norms_2019.csv
#
# Specifications (both predictors; HC1 robust SEs for OLS):
#   (1) all terms, bivariate                        (the association in Fig. 3)
#   (2) terms with >= 1 face, bivariate
#   (3) (2) + log(1 + faces)
#   (4) (3) + log frequency (SUBTLEX 'freq' column) + classifier confidence
#   (5) (4) + dominant part-of-speech fixed effects
#   (6) (5) on terms with >= 10 faces
#   (7) quasi-binomial GLM of male/female face counts, controls as in (5)
#
# Usage:  python si_08_image_regressions.py   (paths via paths.py / env vars GC_IMAGE_FILE,
#         GC_LANC_FILE, GC_RESULTS_DIR)
# Requires pandas, numpy, scipy, statsmodels.
# ==============================================================================
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IMAGE, LANC, results

# ================================== CONFIG ====================================
IMAGE_FILE = IMAGE
LANC_FILE  = LANC
OUT_DIR    = results("si_08")
PERCEPTUAL = ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"]
# ==============================================================================

# NB: keep_default_na=False so that the search terms "none", "null" and "nan"
# are read as words rather than as missing values.
im = pd.read_csv(IMAGE_FILE, low_memory=False, keep_default_na=False,
                 na_values={"Dom_Pos": ["", "NA"], "frac_males": ["NA", ""], "frac_females": ["NA", ""], "avg_age": ["NA", ""]})
lan = pd.read_csv(LANC_FILE, keep_default_na=False)
for c in PERCEPTUAL: lan[c] = pd.to_numeric(lan[c])
lan["lanc"] = lan[PERCEPTUAL].mean(axis=1); lan["key"] = lan.Word.str.lower().str.strip(); im["key"] = im["query"].str.lower().str.strip()
d = im.merge(lan[["key", "lanc"]].drop_duplicates("key"), on="key", how="left")
for c in ["num_faces", "num_males", "num_females", "frac_male", "conc", "freq", "gender_conf"]: d[c] = pd.to_numeric(d[c])
d["logfaces"] = np.log1p(d.num_faces); d["pos"] = d.Dom_Pos.fillna("Unknown").replace({"0": "Unknown"})
print("N conc", d.conc.notna().sum(), "N lanc", d.lanc.notna().sum(), "zero-face", (d.num_faces == 0).sum(),
      "share", round((d.num_faces == 0).mean(), 4), "total faces", int(d.num_faces.sum()))

out = {}
def corr(x, y):
    m = x.notna() & y.notna(); x = x[m]; y = y[m]; n = len(x); r, p = stats.pearsonr(x, y)
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); rho, pr = stats.spearmanr(x, y)
    return dict(n=int(n), r=float(r), lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)),
                t=float(r * np.sqrt((n - 2) / (1 - r * r))), p=float(p), rho=float(rho), prho=float(pr))
f = d[d.num_faces >= 1]
for pred in ["conc", "lanc"]:
    out[pred] = {"corr_all": corr(d[pred], d.frac_male), "corr_face1": corr(f[pred], f.frac_male),
                 "corr_face10": corr(d[d.num_faces >= 10][pred], d[d.num_faces >= 10].frac_male)}
    def fit(formula, data, glm=False):
        g = data.dropna(subset=[pred])
        # HC1 (not HC3) robust SEs: HC3 breaks on a singleton POS category
        res = (smf.glm(formula, g, family=sm.families.Binomial(), var_weights=g.num_faces).fit(scale="X2") if glm
               else smf.ols(formula, g).fit(cov_type="HC1"))
        ci = res.conf_int().loc[pred]
        return dict(b=float(res.params[pred]), se=float(res.bse[pred]), stat=float(res.tvalues[pred]), p=float(res.pvalues[pred]),
                    lo=float(ci[0]), hi=float(ci[1]), n=int(res.nobs))
    full = " + logfaces + freq + gender_conf + C(pos)"
    out[pred]["specs"] = {
        "1_all_biv":               fit(f"frac_male ~ {pred}", d),
        "2_face1_biv":             fit(f"frac_male ~ {pred}", f),
        "3_face1_faces":           fit(f"frac_male ~ {pred} + logfaces", f),
        "4_face1_faces_freq_conf": fit(f"frac_male ~ {pred} + logfaces + freq + gender_conf", f),
        "5_face1_full":            fit(f"frac_male ~ {pred}" + full, f),
        "6_face10_full":           fit(f"frac_male ~ {pred}" + full, d[d.num_faces >= 10]),
        "7_quasibinom_full":       fit(f"I(num_males/num_faces) ~ {pred}" + full, f, glm=True),
    }
    # ventiles (20 equal-count bins) and quintiles of the predictor
    g = d.dropna(subset=[pred]).copy(); g["v"] = pd.qcut(g[pred].rank(method="first"), 20, labels=False)
    out[pred]["vent_all"] = [float(g[g.v == 0].frac_male.mean()), float(g[g.v == 19].frac_male.mean())]
    gf = g[g.num_faces >= 1]; out[pred]["vent_face1"] = [float(gf[gf.v == 0].frac_male.mean()), float(gf[gf.v == 19].frac_male.mean())]
    g["q"] = pd.qcut(g[pred].rank(method="first"), 5, labels=False)
    out[pred]["pzero_q"] = [float((g[g.q == 0].num_faces == 0).mean()), float((g[g.q == 4].num_faces == 0).mean())]
out["anyface_conc"] = corr(d.conc, (d.num_faces > 0).astype(float)); out["anyface_lanc"] = corr(d.lanc, (d.num_faces > 0).astype(float))
out["meta"] = dict(n_terms=int(len(d)), n_lanc=int(d.lanc.notna().sum()), n_zero=int((d.num_faces == 0).sum()), n_face1=int((d.num_faces >= 1).sum()),
                   total_faces=int(d.num_faces.sum()), male_share_pooled=float(d.num_males.sum() / d.num_faces.sum()), mean_fm_face1=float(f.frac_male.mean()))

# ------------------------------------------------------------------ LaTeX rows -----------------------------------
def num(x, dgt=4):
    s = f"{abs(x):.{dgt}f}"; return f"$-{s}$" if x < 0 else f"${s}$"
def fp(p):
    if p < 2.2e-16: return r"$<2.2\times10^{-16}$"
    if p < 0.001:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        if round(m, 1) >= 10: m, e = m / 10, e + 1
        return rf"${m:.1f}\times10^{{{e}}}$"
    return f"${p:.3f}$"
LABELS = [("1_all_biv", r"(1) All terms; terms without faces set to zero (as in Fig.~3); no controls"),
          ("2_face1_biv", r"(2) Terms with $\geq$1 face; no controls"),
          ("3_face1_faces", r"(3) As (2), controlling for the number of faces (log)"),
          ("4_face1_faces_freq_conf", r"(4) As (3), plus word frequency (log) and classifier confidence"),
          ("5_face1_full", r"(5) As (4), plus part-of-speech fixed effects"),
          ("6_face10_full", r"(6) As (5), terms with $\geq$10 faces only"),
          ("7_quasibinom_full", r"(7) Quasi-binomial model of male versus female face counts; controls as in (5)")]
rows = []
for k, lab in LABELS:
    cells = []
    for pred in ["conc", "lanc"]:
        s = out[pred]["specs"][k]
        cells += [num(s["b"]), num(s["se"]), num(s["stat"], 2), fp(s["p"]), f"{s['n']:,}".replace(",", "{,}")]
    rows.append("    " + lab + " & " + " & ".join(cells) + r" \\[2pt]")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(out, open(os.path.join(OUT_DIR, "fig3_image_stats.json"), "w"), indent=1)
open(os.path.join(OUT_DIR, "tab_image_rows.tex"), "w").write("\n".join(rows) + "\n")
for pred in ["conc", "lanc"]:
    print(f"\n== {pred}")
    for k, v in out[pred].items():
        if k == "specs":
            for s, vv in v.items(): print(f"   {s:26s} b={vv['b']:+.4f} se={vv['se']:.4f} stat={vv['stat']:+.2f} P={vv['p']:.2g} n={vv['n']:,}")
        else: print("  ", k, v)
print("\n", out["meta"], "\nwritten to", OUT_DIR)
