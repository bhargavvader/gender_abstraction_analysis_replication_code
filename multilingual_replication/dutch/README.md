# Dutch replication of the male abstraction bias

Replicates the Fig. 1a text analysis of *The Male Abstraction Bias in Internet Culture and Large
Language Models* in Dutch: a gender axis is built from seed pairs in Dutch-trained word vectors
(Eq. S1 of the supplement — the mean over seed pairs of unit(male) − unit(female), re-normalised),
every rated word is scored by its cosine with the axis (positive = masculine), and the score is
correlated with human concreteness ratings. A negative correlation means that more masculine words
are more abstract. This folder reproduces Supplementary Section "Replication of the text-based
result in Dutch" (with its methods subsection "Dutch replication of the text-based analysis"), its
two Supplementary Tables and Supplementary Fig. (`figS_dutch_replication`).

## Contents

| file / folder | what it is |
|---|---|
| `dutch_replication.ipynb` | The analysis, executed, with outputs shown. |
| `dutch_replication.py` | The same code as a script (`python dutch_replication.py`); produces identical files. |
| `nl-brysbaert-2014.tsv` | The norms: Brysbaert, Stevens, De Deyne, Voorspoels & Storms (2014), concreteness (1–5) and age-of-acquisition ratings for Dutch words (columns `word`, `age of acquisition`, `concreteness`). 25,888 rows; 25,881 distinct words with a concreteness rating. |
| `results_nl/` | Outputs (see below). |
| `si_tables.py` | Emits the LaTeX bodies of the two Supplementary Tables from `results_nl/` (`python si_tables.py`). |
| `requirements.txt` | Python packages. |

## Data provenance

The norms file is the copy redistributed, unmodified, in the subs2vec package of van Paridon &
Thompson (2021; `subs2vec/datasets/norms/nl-brysbaert-2014.tsv`,
https://github.com/jvparidon/subs2vec). The authors' own release of the norms has 30,070 rows and is
available from the Ghent University Center for Reading Research
(https://crr.ugent.be) and, as `Brysbaert2014a.csv`, from the ManyNorms / semanticprimeR release
(https://github.com/SemanticPriming/semanticprimeR/releases/tag/v0.0.1). The results reported in
the supplement are based on the subs2vec copy; the code runs unchanged on the full file (rename it
to `nl-brysbaert-2014.tsv` or edit `NORMS`), which changes the numbers slightly because of the
additional words.

* Brysbaert, M., Stevens, M., De Deyne, S., Voorspoels, W. & Storms, G. (2014). Norms of age of
  acquisition and concreteness for 30,000 Dutch words. *Acta Psychologica* 150, 80–84.
  https://doi.org/10.1016/j.actpsy.2014.04.010
* van Paridon, J. & Thompson, B. (2021). subs2vec: word embeddings from subtitles in 55 languages.
  *Behavior Research Methods* 53, 629–655.

## Word vectors

The static vector table of spaCy's `nl_core_news_lg` pipeline, **version 3.8.0** (500,000 entries ×
300 dimensions; fastText CBOW vectors trained by Explosion on the Dutch portions of OSCAR Common
Crawl and Wikipedia; Honnibal, Montani, Van Landeghem & Boyd, spaCy, https://doi.org/10.5281/zenodo.1212303).
It is a pip-installable model, not a data file:

```
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_lg-3.8.0/nl_core_news_lg-3.8.0-py3-none-any.whl
```

Words outside the model's vocabulary are not scored; 23,758 of the 25,881 rated words (91.8 %) have
a vector. Word frequency for the frequency control comes from the `wordfreq` library (Speer, 2022;
Zipf-scaled Dutch frequencies; 24,101 of the rated words have an entry).

## Running it

```
jupyter notebook dutch_replication.ipynb      # run all cells
# or
python dutch_replication.py
```

Both read `nl-brysbaert-2014.tsv` from the folder they are in and write `results_nl/`. Runtime is
about a minute. Results were produced with Python 3.12, spaCy 3.8, wordfreq 3.1, numpy 2.4, pandas 3.0,
scipy 1.17, statsmodels 0.15 and matplotlib 3.10; the computation involves no randomness.

## Design choices (shared with the other language replications)

* **Seed pairs.** Primary (chosen before any result was seen): man/vrouw, jongen/meisje,
  mannelijk/vrouwelijk, vader/moeder, broer/zus — the paper's five English pairs with the two
  pronoun pairs replaced by kinship pairs, because Dutch *zij* also means *they* and *haar* also
  means *hair*. Extended: the primary pairs plus zoon/dochter, oom/tante, koning/koningin, heer/dame,
  opa/oma. Pronoun set, reported to show the consequence of the ambiguity: man/vrouw, hij/zij,
  hem/haar, mannelijk/vrouwelijk, jongen/meisje.
* **Held-out check** (`validation_nl.csv`). 27 gendered pairs not used as primary seeds
  (koning/koningin, acteur/actrice, meneer/mevrouw, mannen/vrouwen, …); a valid axis orders the male
  above the female member. Primary axis: 27/27.
* **Specifications** (one row each in `stats_nl.csv`, for each seed set): all words; excluding
  words that contain a gendered morpheme or a kinship/role stem together with the seed words;
  excluding the 5 % and 10 % of words with the most extreme gender scores; HC1-robust standardised β
  with no controls, with log frequency, with length (characters), with both, and with frequency,
  length and age of acquisition; Pearson's r within each quintile of word frequency.
* **Seed-pair sensitivity** (`pole_sensitivity_nl.csv`): leave-one-pair-out on the primary set,
  every single pair, the pronoun set and its pronoun pairs alone, and the first principal component
  of the ten unit-normalised pair differences of the extended set (uncentred SVD, sign-aligned with
  their mean), each with the eight most feminine and eight most masculine words.
* **Deciles** (`deciles_nl.csv`, Supplementary Fig.): mean concreteness within deciles of the gender
  score, computed separately for each axis.

## Headline numbers (results_nl/stats_nl.csv, "all terms")

| axis | r | 95 % CI | n |
|---|---|---|---|
| primary (5 pairs) | −0.268 | −0.279 to −0.256 | 23,758 |
| extended (10 pairs) | −0.292 | −0.304 to −0.280 | 23,758 |
| pronoun set (invalid in Dutch: *zij* = they, *haar* = hair) | +0.217 | 0.205 to 0.229 | 23,758 |

## Output columns

* `stats_nl.csv` — `seeds` (primary / extended / pronoun), `specification`, `r` (Pearson's r, or the
  standardised β for OLS rows), `ci_lo`, `ci_hi` (95 % CI), `p` (two-tailed), `rho` (Spearman; blank
  for OLS rows), `n`.
* `deciles_nl.csv` — `decile` (1 = most feminine, 10 = most masculine), `mean` concreteness, `count`, `seeds`.
* `pole_sensitivity_nl.csv` — `specification`, `n_pairs`, `r`, `ci_lo`, `ci_hi`, `p`, `n`, `most_female`, `most_male`.
* `scores_nl.csv` — `word`, `conc`, and the gender score of every rated word under each axis
  (`gender::primary`, `gender::extended`, `gender::pronoun`; blank where the word has no vector).
* `validation_nl.csv` — each held-out pair with both scores and whether the male member scores higher, per axis.
* `seeds_nl.json` — the seed pairs found in the vectors for each axis.
