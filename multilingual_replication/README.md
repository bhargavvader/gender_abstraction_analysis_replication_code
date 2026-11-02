# Multilingual text-analysis replication

Language-specific replications of the association between embedding gender scores and human concreteness ratings. Each analysis builds a gender direction from language-specific seed pairs, scores rated words, and reports correlations and seed/control sensitivities. Positive gender scores denote the masculine direction; a negative correlation denotes greater male association for more abstract words.

## Contents

| Folder | Analysis and supplied inputs |
|---|---|
| `dutch/` | Python script and notebook; Brysbaert et al. (2014) norms; saved `results_nl/`; table exporter |
| `chinese/` | Python script and notebook; Chan & Tse (2024) and Song & Li (2021) norms; saved `results_zh/` and `results_zh_song/`; table exporter |
| `estonian/` | CLI script and notebook; Proos & Aigro (2023) data and standardized copy; saved `results_et/` |
| `persian/` | Notebook; Nemati et al. (2026) human best–worst norms and a machine-extrapolated secondary set; saved `results_fa/` |
| `forest/` | Notebook for Portuguese, Spanish, French, Indonesian, Italian and Turkish; `norms/` and saved `results_forest/` |

Language READMEs and `forest/norms/NORMS_README.md` retain supplied source references. The norms README also describes Croatian files and an `original/` folder that were not supplied here. German and Serbian norm files are included, but the forest notebook does not configure analyses for them. Croatian replication code/data were not supplied; its forest row is optional.

Saved results are supplied outputs, not newly recomputed results. Large models/vectors are excluded. Norm files and result tables can be inspected without downloading models.

## Packages and models to download

Use Python 3.10 or newer. Install the common packages with:

```bash
pip install -r multilingual_replication/requirements.txt
```

Run notebooks with their working directory set to the language folder (for example, `cd multilingual_replication/persian` before starting Jupyter). Their `BASE = Path.cwd()` configuration expects this. Dutch and Chinese scripts locate their own directories automatically.

| Analysis | Required external vectors/model | Where to place or install |
|---|---|---|
| Dutch | spaCy `nl_core_news_lg`, version **3.8.0** | Install model wheel below; uses its static vector table |
| Mandarin Chinese | spaCy `zh_core_web_lg`, version **3.8.0** | Install model wheel below; uses its static vector table |
| Estonian | fastText Common Crawl/Wikipedia `cc.et.300.vec` (or `.vec.gz` for CLI), or `cc.et.300.bin` | Notebook: in `estonian/`; CLI accepts an explicit vector/model path |
| Persian | fastText `cc.fa.300.vec`, decompressed text vectors | `persian/cc.fa.300.vec` |
| Forest plot | fastText `cc.pt.300.vec`, `cc.es.300.vec`, `cc.fr.300.vec`, `cc.id.300.vec`, `cc.it.300.vec`, `cc.tr.300.vec` | In `forest/`, decompressed |

```bash
pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_lg-3.8.0/nl_core_news_lg-3.8.0-py3-none-any.whl
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-3.8.0/zh_core_web_lg-3.8.0-py3-none-any.whl
```

The fastText files are available from the [official Common Crawl vector downloads](https://fasttext.cc/docs/en/crawl-vectors.html). Download the language-specific **text vectors** for the notebook routes, decompress `.vec.gz`, and retain the filenames above. Files can be several GB each; allow sufficient disk space. The text routes stream needed vectors and do not require the fastText Python package.

The Estonian binary route additionally requires `pip install fasttext-wheel`. If neither `--vec` nor `--model` is passed, the Estonian CLI attempts a model download automatically; pass a local path to avoid this. Binary subword OOV inference and vocabulary-only text lookup are different specifications; use `--oov skip` for the vocabulary-only route.

## Run examples

From the repository root:

```bash
python multilingual_replication/dutch/dutch_replication.py
python multilingual_replication/chinese/chinese_replication.py
python multilingual_replication/estonian/estonian_replication.py \
  --norms multilingual_replication/estonian/data.csv \
  --vec multilingual_replication/estonian/cc.et.300.vec \
  --oov skip --out multilingual_replication/estonian/results_et
```

For Persian, open `persian_replication.ipynb` from within `persian/` and run its cells in order. For the forest analysis, open `forest_plot_replication.ipynb` from within `forest/`. Missing language vector files are skipped, so a partial run will contain fewer languages. Download all six files for a complete smaller-norm analysis. The forest notebook includes fixed summary rows for the earlier large-norm analyses and optionally reads Persian/Croatian result files; it does not rerun those analyses. The English row is an across-model summary, with a different interval interpretation from a single-language Fisher interval.

Scripts/notebooks write to their language's `results_*` folder and overwrite corresponding saved outputs. Retain a copy if comparing a fresh run with the supplied results. Forest figures/tables describe the subset actually available during that run.

## Packaging checks

All supplied Python scripts and notebook code cells passed syntax checks. Norms, model filenames, result paths, seed-axis logic and output structure were inspected. Notebook configuration was changed from machine-specific paths to the folder layout above. No model downloads or full language analyses were performed during this packaging pass; saved outputs were retained. Language-specific READMEs may describe earlier packaging states; use this README for the current layout and download instructions.
