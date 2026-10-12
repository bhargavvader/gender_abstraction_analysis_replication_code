#!/usr/bin/env python3
"""figure_05_compute.py

Computes the data behind Fig. 5: for each (dataset, generator, scorer) cell, the Pearson correlation between
the human-rated concreteness of each cue word and the frequency-weighted mean gender association of the words
freely associated with it, with each of the 12 language models of registry.py used in turn to score the
responses.

Both gender scores AND cue concreteness come from the master file (conc_brysbaert), so this run is
self-contained in that one source.

Sign convention is verified empirically rather than assumed: for every model, mean(male seed words) > mean(female
seed words), so negative r means "abstract cues elicit more male-typed responses".

Outputs (results/figure_05/)
  master12_correlations.csv      r, p, n_cues, Fisher 95% CI for every cell  -> read by figure_05_plot.py
  master12_binned.csv            per-decile means for every cell
  master12_signcheck.csv

Inputs: the master; SWOW strength_SWOW-EN_R123_20180827.csv (tab-separated, QUOTE_NONE mandatory);
LWOW associations_data_all_scores.json.
"""

import csv, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MASTER, SWOW, LWOW, results
from registry import REGISTRY

OUT = results("figure_05") + "/"

N_BINS = 10
SPECIAL = frozenset({"no more responses", "unknown word", "nan", "", "none", "na"})

# scorer display name -> column in the master file: the 12-model registry shared with Figs. 1, 2 and 4c
SELECTED = dict(REGISTRY)

MALE_SEEDS = ["man", "men", "he", "him", "his", "father", "boy",
              "king", "husband", "son", "brother", "uncle"]
FEMALE_SEEDS = ["woman", "women", "she", "her", "hers", "mother", "girl",
                "queen", "wife", "daughter", "sister", "aunt"]

DATASETS = ["SWOW", "LWOW"]
GENERATORS = {"SWOW": ["Humans"],
              "LWOW": ["Humans", "Llama3", "Haiku", "Mistral"]}
SERIES = [("SWOW", "Humans"), ("LWOW", "Humans"), ("LWOW", "Llama3"), ("LWOW", "Haiku"), ("LWOW", "Mistral")]


# ------------------------------------------------------------------ loading
def load_master():
    m = pd.read_csv(MASTER)
    m["w"] = m["term"].astype(str).str.lower().str.strip()
    # 2,956 terms appear as both 'word' and 'phrase'; .first() takes the first
    # non-null value per column, so a term is never dropped for one model just
    # because its other row happened to be blank.
    cols = list(SELECTED.values()) + ["conc_brysbaert"]
    lut = m.groupby("w")[cols].first()
    return lut


def sign_check(lut):
    rows = []
    for lbl, col in SELECTED.items():
        s = lut[col]
        mv = s.reindex(MALE_SEEDS).dropna()
        fv = s.reindex(FEMALE_SEEDS).dropna()
        rows.append({
            "model": lbl, "column": col,
            "mean_male_seeds": mv.mean(), "mean_female_seeds": fv.mean(),
            "male_minus_female": mv.mean() - fv.mean(),
            "positive_is_male": bool(mv.mean() > fv.mean()),
            "n_male_seeds": len(mv), "n_female_seeds": len(fv),
            "lexicon_coverage": float(s.notna().mean()),
        })
    return pd.DataFrame(rows)


def swow_long():
    df = pd.read_csv(SWOW, sep="\t", quoting=csv.QUOTE_NONE)   # QUOTE_NONE is mandatory
    df["cue"] = df["cue"].astype(str).str.lower().str.strip()
    df["response"] = df["response"].astype(str).str.lower().str.strip()
    df = df[~df["response"].isin(SPECIAL)]
    df = df[df["response"] != df["cue"]]
    return {"Humans": df[["cue", "response", "R123"]].rename(columns={"R123": "count"})}


def lwow_long():
    a = pd.read_json(LWOW)
    a["cue"] = a["cue"].astype(str).str.lower().str.strip()
    cols = {"Humans": "responses_human", "Llama3": "responses_llama3",
            "Haiku": "responses_haiku", "Mistral": "responses_mistral"}
    out = {}
    for gen, col in cols.items():
        cues, resps, counts = [], [], []
        for cue, d in zip(a["cue"], a[col]):
            for k, v in d.items():
                cues.append(cue)
                resps.append(str(k).lower().strip())
                counts.append(v)
        df = pd.DataFrame({"cue": cues, "response": resps, "count": counts})
        out[gen] = df[~df["response"].isin(SPECIAL)]
    return out


# ------------------------------------------------------------------ analysis
def weighted_cue_gender(long_df, score_lut):
    s = long_df["response"].map(score_lut)
    ok = s.notna()
    d = long_df[ok]
    num = (s[ok] * d["count"]).groupby(d["cue"]).sum()
    den = d.groupby("cue")["count"].sum()
    return (num / den).rename("gender")


def bin_and_correlate(gender_by_cue, conc):
    d = pd.DataFrame({"gender": gender_by_cue})
    d["conc"] = d.index.map(conc)
    d = d.dropna()
    if len(d) < 50:
        return None, np.nan, np.nan, len(d), (np.nan, np.nan)
    r, p = pearsonr(d["conc"], d["gender"])
    n = len(d)
    z, se = np.arctanh(r), 1.0 / np.sqrt(len(d) - 3)
    ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
    d["bin"] = pd.qcut(d["conc"], q=N_BINS, labels=False, duplicates="drop") + 1
    agg = d.groupby("bin").agg(
        conc_mean=("conc", "mean"),
        gender_mean=("gender", "mean"),
        gender_sem=("gender", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=("gender", "size"),
    ).reset_index()
    return agg, r, p, n, ci


# ------------------------------------------------------------------ main
def main():
    lut = load_master()
    conc = lut["conc_brysbaert"]

    sc_df = sign_check(lut)
    sc_df.to_csv(OUT + "master12_signcheck.csv", index=False)

    long_data = {"SWOW": swow_long(), "LWOW": lwow_long()}

    binned, corr = [], []
    for lbl, col in SELECTED.items():
        score_lut = lut[col]
        for ds in DATASETS:
            for gen in GENERATORS[ds]:
                cue_g = weighted_cue_gender(long_data[ds][gen], score_lut)
                agg, r, p, n, ci = bin_and_correlate(cue_g, conc)
                corr.append({"dataset": ds, "generator": gen, "scorer": lbl,
                             "r": r, "p": p, "n_cues": n,
                             "ci_low": ci[0], "ci_high": ci[1]})
                if agg is not None:
                    t = agg.copy()
                    t.insert(0, "dataset", ds)
                    t.insert(1, "generator", gen)
                    t.insert(2, "scorer", lbl)
                    binned.append(t)

    corr_df = pd.DataFrame(corr)
    corr_df.to_csv(OUT + "master12_correlations.csv", index=False)
    pd.concat(binned, ignore_index=True).to_csv(OUT + "master12_binned.csv", index=False)

    # ---------------------------------------------------------- console
    mean_r = corr_df.groupby("scorer")["r"].mean().sort_values()          # x-axis order of Fig. 5
    piv = corr_df.pivot_table(index=["dataset", "generator"],
                              columns="scorer", values="r")[mean_r.index]
    print(piv.round(3).to_string())
    print(f"\nnegative r in {(corr_df['r'] < 0).sum()}/{len(corr_df)}; "
          f"CI excludes zero in "
          f"{((corr_df['ci_low'] < 0) & (corr_df['ci_high'] < 0)).sum()}/{len(corr_df)}")
    print("\nsign check (male seeds minus female seeds, all should be > 0):")
    print(sc_df[["model", "male_minus_female", "positive_is_male"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
