# Chinese (Mandarin) replication of the male abstraction bias

Replicates the Fig. 1a text analysis of *The Male Abstraction Bias in Internet Culture and Large
Language Models* in Chinese: a gender axis is built from seed pairs in Chinese-trained word vectors
(Eq. S1 of the supplement — the mean over seed pairs of unit(male) − unit(female), re-normalised),
every rated two-character word is scored by its cosine with the axis (positive = masculine), and the
score is correlated with human concreteness ratings. A negative correlation means that more masculine
words are more abstract. This folder reproduces Supplementary Section "Replication of the text-based
result in Chinese", its two Supplementary Tables and Supplementary Fig. (`figS_chinese_replication`).

## Contents

| file / folder | what it is |
|---|---|
| `chinese_replication.ipynb` | The analysis, executed, with outputs shown. |
| `chinese_replication.py` | The same code as a script (`python chinese_replication.py`); produces identical files. |
| `zh-chan-tse-2024.csv` | Main norms: Chan & Tse (2024), 25,281 two-character words rated for concreteness (1–7), familiarity, imageability, valence and arousal by Cantonese-speaking participants in Hong Kong; each word in traditional and simplified characters (`word_chinese_traditional`, `word_chinese_simplified`). The simplified forms are analysed (25,044 distinct). Unmodified copy of `Chan2024.csv`. |
| `Chan2024.csv` | The same file under its original name. |
| `zh-song-li-2021.csv` | Check norms: Song & Li (2021), 3,783 simplified two-character words rated for concreteness (1–7) by Mainland Mandarin speakers. Unmodified copy of `Song2021.csv`. |
| `Song2021.csv` | The same file under its original name. |
| `results_zh/` | Outputs for the Chan & Tse norms (the reported result). |
| `results_zh_song/` | Outputs for the Song & Li norms (the Mainland check). |
| `si_tables.py` | Emits the LaTeX bodies of the two Supplementary Tables from `results_zh/` (`python si_tables.py zh results_zh fig:S-chinese`). |
| `requirements.txt` | Python packages. |

Both norm files come from the ManyNorms / semanticprimeR release
(`https://github.com/SemanticPriming/semanticprimeR/releases/tag/v0.0.1`, files `Chan2024.csv` and
`Song2021.csv`), which redistributes the published norms:

* Chan, Y.-L. & Tse, C.-S. (2024). Decoding the essence of two-character Chinese words: unveiling
  valence, arousal, concreteness, familiarity, and imageability through word norming.
  *Behavior Research Methods* 56, 7574–7601. https://doi.org/10.3758/s13428-024-02437-w
* Song, D. & Li, D. (2021). Psycholinguistic norms for 3,783 two-character words in simplified
  Chinese. *SAGE Open* 11. https://doi.org/10.1177/21582440211054495

## Word vectors

The static vector table of spaCy's `zh_core_web_lg` pipeline, **version 3.8.0** (500,000 entries ×
300 dimensions; fastText CBOW vectors trained by Explosion on the Chinese portions of OSCAR Common
Crawl and Wikipedia; Honnibal, Montani, Van Landeghem & Boyd, spaCy, https://doi.org/10.5281/zenodo.1212303).
It is not a data file but a pip-installable model:

```
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-3.8.0/zh_core_web_lg-3.8.0-py3-none-any.whl
```

Words outside the model's vocabulary are not scored; 23,961 of the 25,044 Chan & Tse words (95.7 %)
and 3,782 of the 3,783 Song & Li words have a vector.

## Running it

```
jupyter notebook chinese_replication.ipynb      # run all cells
# or
python chinese_replication.py
```

Both read the two norm files from the folder they are in and write `results_zh/` and
`results_zh_song/`. Runtime is about a minute. Results were produced with Python 3.12, spaCy 3.8,
numpy 2.4, pandas 3.0, scipy 1.17, statsmodels 0.15 and matplotlib 3.10; the computation involves no
randomness, so any recent versions give the same numbers.

## Design choices (as in the other language replications)

* **Seed pairs.** Primary: 男人/女人 (man/woman), 男孩/女孩 (boy/girl), 男性/女性 (male/female),
  父亲/母亲 (father/mother), 哥哥/姐姐 (elder brother/elder sister) — the counterparts of the Dutch
  and Estonian primary sets. Extended: the primary pairs plus 儿子/女儿, 叔叔/阿姨, 国王/女王, 先生/女士,
  爷爷/奶奶. Pronoun variant: 他/她 in place of 哥哥/姐姐 (unambiguous in written Chinese, unlike Dutch
  *zij*/*haar*).
* **Held-out check.** 18 gendered pairs never used as seeds (丈夫/妻子, 新郎/新娘, 王子/公主, 皇帝/皇后, …);
  a valid axis orders the male above the female member. Primary axis: 18/18; extended and pronoun: 17/18.
* **Specifications** (one row each in `stats_*.csv`, for each seed set): all words; excluding words
  that contain a gender-marking character (男 女 父 母 爸 妈 哥 姐 … 他 她) together with the seed
  words; excluding the 5 % and 10 % of words with the most extreme gender scores; words with rater
  s.d. of concreteness at or below the median (Chan & Tse only); HC1-robust standardised β with and
  without log word frequency (the `frequency_log_word` column of Chan & Tse; log(1 + `frequency_subtitle`)
  for Song & Li). Length is not controlled because every item is a two-character word.
* **Seed-pair sensitivity** (`pole_sensitivity_*.csv`): leave-one-pair-out on the primary set, every
  single pair, the pronoun set, and the first principal component of the ten unit-normalised pair
  differences of the extended set (uncentred SVD, sign-aligned with their mean), each with the eight
  most feminine and eight most masculine words.
* **Deciles** (`deciles_*.csv`, Supplementary Fig.): mean concreteness within deciles of the gender
  score, computed separately for each axis.

## Headline numbers (results_zh/stats_zh.csv, "all terms")

| axis | r | 95 % CI | n |
|---|---|---|---|
| primary (5 pairs) | −0.111 | −0.123 to −0.098 | 23,961 |
| extended (10 pairs) | −0.193 | −0.205 to −0.181 | 23,961 |
| pronoun variant | −0.168 | −0.180 to −0.155 | 23,961 |

Song & Li (Mainland Mandarin, n = 3,782): −0.189, −0.226 and −0.288 for the same three axes.

## Output columns

* `stats_*.csv` — `seeds` (primary / extended / pronoun), `specification`, `r` (Pearson's r, or the
  standardised β for OLS rows), `ci_lo`, `ci_hi` (95 % CI), `p` (two-tailed), `rho` (Spearman; blank
  for OLS rows), `n`, `validation_accuracy`, `validation_pairs`.
* `deciles_*.csv` — `decile` (1 = most feminine, 10 = most masculine), `mean` concreteness, `count`, `seeds`.
* `pole_sensitivity_*.csv` — `specification`, `n_pairs`, `r`, `ci_lo`, `ci_hi`, `p`, `n`, `most_female`, `most_male`.
* `scores_*.csv` — `word`, `conc`, and the gender score of every rated word under each axis
  (`gender::primary`, `gender::extended`, `gender::pronoun`; blank where the word has no vector).
* `seeds_*.json` — the seed pairs actually found in the vectors for each axis.
