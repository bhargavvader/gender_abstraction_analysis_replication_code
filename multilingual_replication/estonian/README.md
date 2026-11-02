# Cross-linguistic replication of the male abstraction bias (text analysis)

Replicates the Fig. 1a test — do terms with more masculine embedding associations carry
lower human concreteness ratings? — with language-specific embeddings and
language-specific human concreteness norms. Dutch has been run end-to-end here;
Estonian and Mandarin are configured and run with one command once the norm files
(and, for Estonian, a vector file) are placed in `data/`.

Method follows the paper's SI 1.2: unit-normalise seed-pair embeddings, average the
male-minus-female differences, re-normalise, and score each rated term by its cosine
with that axis (positive = masculine). Statistics follow SI 3 and 5: Pearson r with
Fisher 95% CI, Spearman ρ, gender-decile means, HC1-robust standardised β with
frequency / length / AoA controls, within-frequency-quintile correlations, and
exclusion of lexically gendered terms and of the 5–10% most gendered terms.

## Dutch result

Norms: Brysbaert, Stevens, De Deyne, Voorspoels & Storms (2014), 25,883 rated words
(copy in `data/nl-brysbaert-2014.tsv`, taken from the subs2vec repository's bundled norms).
Model: spaCy `nl_core_news_lg` static vectors (500k × 300, Dutch-trained). (ConceptNet
Numberbatch was also run and was near-null, r = −0.03, as ConceptNet is for English single
words in the paper; it is commented out in `CONFIG['nl']` and not reported.)

**Pre-specified primary pole set** (chosen before any result was seen, mirroring the
paper's five pairs where the Dutch item is unambiguous): man/vrouw, jongen/meisje,
mannelijk/vrouwelijk, vader/moeder, broer/zus. The paper's two pronoun pairs were not
used because *zij* also means *they* and *haar* also means *hair*; both were kept as a
labelled "pronoun" robustness set.

| Model | Spec. | r | 95% CI | ρ | n |
|---|---|---|---|---|---|
| spaCy nl | all terms | **−0.268** | −0.279, −0.256 | −0.249 | 23,758 |
| spaCy nl | excl. lexically gendered terms + seeds | −0.268 | −0.280, −0.256 | −0.246 | 23,062 |
| spaCy nl | excl. 10% most gendered terms | −0.284 | −0.296, −0.271 | −0.264 | 21,382 |
| spaCy nl | β, + log frequency (wordfreq zipf) | −0.269 | −0.281, −0.256 | | 23,407 |
| spaCy nl | β, + frequency + length | −0.231 | −0.243, −0.218 | | 23,407 |
| spaCy nl | β, + frequency + length + AoA | −0.245 | −0.258, −0.231 | | 23,398 |
| spaCy nl | least-frequent quintile / most-frequent quintile | −0.236 / −0.233 | | | 4,682 each |
| spaCy nl, 10-pair extended set | all terms | −0.292 | −0.304, −0.280 | −0.292 | 23,758 |

Decile means (spaCy, primary set): most-feminine decile 3.57 → most-masculine decile
2.74 on the 1–5 scale (cf. 3.42 → 2.83 averaged across models in English, Fig. 1a).
Every frequency quintile is negative (−0.23 to −0.30).

### Pole sensitivity (Fig. panel B; `results/pole_sensitivity_nl_*.csv`)

Dutch static vectors are more sensitive to the seed specification than the English models:

* Every axis built from **several** unambiguous kinship/role pairs is negative:
  primary set −0.27; extended 10-pair set −0.29; leave-one-pair-out −0.15 to −0.34;
  first principal component of the 10 pair-differences (43% of variance, all pairs load
  0.38–0.84) −0.22.
* Single pairs are noisy: 6 of 10 kinship/role pairs are negative (heer/dame −0.38 …
  opa/oma −0.14), four are near zero or positive (oom/tante +0.03, mannelijk/vrouwelijk
  +0.04, zoon/dochter +0.12, man/vrouw +0.24). The man/vrouw axis's male extreme is
  colloquial 'guy' insults (prutser, malloot, mafkees, gozer), i.e. *man* as an informal
  address term rather than a gender referent.
* The **pronoun pairs flip the sign** (hij/zij +0.36, hem/haar +0.21, five-pair pronoun
  set +0.22). Their axes are visibly not gender axes: the "female" extreme of hij/zij is
  wij, we, hun, zij, gezamenlijk (plural/collective), and of hem/haar is van, hun, alsook,
  rondom (possessive/function words). This is the expected consequence of *zij* = they and
  *haar* = hair/her, and is why pronoun poles should not be used in Dutch.

Recommendation for the paper: report the primary-set result with the leave-one-out range
and the PC1 axis as robustness, note the pronoun ambiguity explicitly, and add
fastText `cc.nl.300` plus the multilingual transformer embeddings (Qwen3, Nomic v2 MoE,
text-embedding-3-large, which already sit in the 12-model registry) locally to check
whether the single-pair instability is specific to spaCy's static table.

## Supplementary-material files (`tex/`)

* `tex/supplementary_methods.tex` — the uploaded file plus a new subsection
  `sec:dutch-methods` ("Dutch replication of the text-based analysis") inserted at the end of
  *Data and methods*, before the image face-count section. Pure insertion; nothing else changed.
* `tex/supplementary_results.tex` — the uploaded file plus a new final section `si:dutch`
  ("Replication of the text-based result in Dutch") with two tables (`tab:S-dutch-spec`,
  `tab:S-dutch-poles`) and one figure (`fig:S-dutch`, file `figS_dutch_replication`). Appended
  at the end so that existing section/table/figure numbers are unchanged. Tables use
  `>{\raggedright\arraybackslash}p{}` columns (package `array`).
* `tex/references.bib` — the project bibliography with a new block appended at the end
  holding the seven keys the SI files cite but the file lacked: `kuperman2012` and
  `warriner2013` (Supplementary Section 3, previously only in
  `references_additions_si_text.bib`) and `proos2023`, `xu2020`, `bolukbasi2016`,
  `honnibal2020spacy`, `speer2022wordfreq` (Dutch replication). `brysbaert2014dutch`,
  `meta2024llama32` and `grattafiori2024llama3` already existed and were left untouched.
  Every key cited in the two SI files now resolves; BibTeX runs with no warnings.
* `make_si_dutch_tables.py` — emits both table bodies from the results CSVs (no hand-typed
  numbers); `tex/results_section_template.tex` + `tex/build_results_section.py` assemble the
  section.
* `figS_dutch_replication.py` — the single-panel supplementary figure.

## Files

* `xling_replication.py` — the pipeline (`--lang nl|et|zh`, `--norms`, `--freq`). All
  language-specific choices are in `CONFIG`.
* `pole_sensitivity.py <lang> "<model name>"` — seed-pair sensitivity table, incl. PC1 axis.
* `make_figure.py <lang>` — exploratory two-panel figure (decile plot + sensitivity forest plot).
* `results/stats_nl.csv`, `results/deciles_nl.csv`, `results/scores_nl.csv`
  (per-word gender scores), `results/seeds_nl.json`, `results/pole_sensitivity_nl_*.csv`,
  `results/figS_dutch_replication.{png,pdf}` (SI figure), `results/fig_xling_nl.{png,pdf}` (exploratory).
* Diagnostic scripts used while choosing poles: `diagnose_poles_nl.py`, `pole_agreement_nl.py`, `pole_pca_nl.py`.

## Running Estonian and Mandarin (locally)

The sandbox could not reach OSF, figshare, fastText's CDN or the Tartu/EKI servers, so
these two need files dropped into `data/`:

**Estonian** — use the self-contained `estonian_replication.py` (below); the notes that
follow describe the inputs it expects.
1. Norms: Proos & Aigro (2023), OSF project `y4bn5` → save as `data/et-proos-aigro-2023.csv`.
   Ratings are on a continuous 0–10 scale (not 1–5); correlations are unaffected. Homonyms
   are rated once per sense in that file — pass `--dup mean` to average them per word form.
   If the columns are not named `word` / `concreteness`, pass `--word-col` / `--conc-col`.
2. Vectors: fastText `cc.et.300.vec.gz` (fasttext.cc, Common Crawl + Wikipedia) or EstNLTK's
   `word2vec_words.sg.s200` (exported to .vec text). The full fastText file is 1.2 GB, so
   shrink it locally first with `filter_vectors.py` (keeps only the ~36k norm words plus the
   seed/validation pairs; output ≈ 100 MB, ≈ 35 MB gzipped) and upload that.
3. Frequency: `wordfreq` has no Estonian list; pass `--freq word,freq.csv` if available,
   otherwise the frequency control is skipped and the other specifications still run.
4. Poles: Estonian has no gendered pronouns (*tema/ta*), so the primary set is
   mees/naine, poiss/tüdruk, mehelik/naiselik, isa/ema, vend/õde; a second kinship set
   (poeg/tütar, härra/proua) is included.

```
pip install fasttext-wheel numpy pandas scipy statsmodels matplotlib
# default: fetches fastText's official Estonian model through the fastText package
#   (fasttext.util.download_model('et') -> cc.et.300.bin, ~7 GB on disk, ~8 GB RAM to load)
python estonian_replication.py --norms data/et-proos-aigro-2023.csv
# lighter alternative: the text vectors (cc.et.300.vec.gz, 1.2 GB) are streamed, not loaded
python estonian_replication.py --norms data/et-proos-aigro-2023.csv --vec cc.et.300.vec.gz
```
Outputs land in `results_et/`: `stats_et.csv`, `pole_sensitivity_et.csv` (leave-one-pair-out,
single pairs, PC1, with the most feminine/masculine words per axis), `deciles_et.csv`,
`scores_et.csv`, `seeds_et.json` and `figS_estonian_replication.{png,pdf}` in the same design
as the Dutch panel. The printed summary includes the held-out validation (20 gendered word pairs
never used as seeds; the axis should order the male member above the female one in nearly all).
Frequency control: word counts from the fastText vocabulary (`.bin`) or file rank (`.vec`).
Both code paths were exercised in the sandbox with stand-in vectors (a treebank-trained
throwaway model and Numberbatch's 4k-term Estonian subspace); those runs are smoke tests only.

**Mandarin**
1. Xu & Li (2020) MELD-SCH ratings (figshare .rar → .xlsx) → `data/MELD-SCH_concreteness.xlsx`;
   set the word/concreteness column names in `CONFIG['zh']['norms']`.
2. Models already installed/extracted here: spaCy `zh_core_web_lg` (simplified script) and
   Numberbatch zh. Numberbatch zh is mostly **traditional** script, so convert the norms
   with `opencc` (s2t) for that model or treat it as secondary. fastText `cc.zh.300` is the
   natural third model locally.
3. Poles: 男人/女人, 男性/女性, 男孩/女孩, 他/她, 父亲/母亲 (他/她 is a written-only
   distinction, so the kinship set 爸爸/妈妈, 儿子/女儿 is provided as the robustness set).
   `wordfreq` covers zh for the frequency control.

```
python xling_replication.py --lang zh --norms data/MELD-SCH_concreteness.xlsx
```

## Environment
Python 3, numpy, pandas, scipy, statsmodels, matplotlib, wordfreq, spacy 3.8 +
`nl_core_news_lg` / `zh_core_web_lg` 3.8.0 (GitHub release wheels); ConceptNet Numberbatch
17.06 from the gensim-data GitHub release, filtered to `/c/nl/`, `/c/et/`, `/c/zh/`
(`data/nb_nl_et_zh.txt`, not included here because of size — regenerate with
`zcat numberbatch-17.06.txt.gz | grep -E '^/c/(nl|et|zh)/'`).
