#!/usr/bin/env python3
"""
Estonian replication of the Fig. 1a text analysis (male abstraction bias), self-contained.

Mirrors the Dutch replication exactly: a gender axis is built from seed pairs in a
language-specific embedding (Eq. S1 of the supplement), every rated word is scored by its
cosine with that axis, and the score is correlated with human concreteness ratings.

Inputs
  --norms   data.csv from the OSF archive of Proos & Aigro (2023), "Concreteness ratings for
            36,000 Estonian words" (columns: lemma, concreteness_index [0-10], concreteness_sd,
            concreteness_ratingcount, ...).
  Vectors (one of):
  (default) fastText's official Estonian model, fetched through the fastText Python package:
            fasttext.util.download_model('et')  ->  cc.et.300.bin  (Common Crawl + Wikipedia,
            2M words, 300-d; ~7 GB on disk, needs ~8 GB RAM to load).
  --model   path to an already-downloaded cc.et.300.bin (skips the download).
  --vec     path to a word2vec/fastText TEXT vector file (.vec or .vec.gz), e.g. cc.et.300.vec.gz
            from https://fasttext.cc/docs/en/crawl-vectors.html; streamed, so RAM stays low.

Outputs (in --out, default results_et/)
  stats_et.csv, pole_sensitivity_et.csv, deciles_et.csv, scores_et.csv, seeds_et.json,
  figS_estonian_replication.png / .pdf, and a printed summary.

Usage
  pip install fasttext-wheel numpy pandas scipy statsmodels matplotlib
  python estonian_replication.py --norms data.csv                      # downloads cc.et.300.bin
  python estonian_replication.py --norms data.csv --model cc.et.300.bin
  python estonian_replication.py --norms data.csv --vec cc.et.300.vec.gz
"""
import argparse, gzip, json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

# ----------------------------------------------------------------------------
# Estonian configuration
# ----------------------------------------------------------------------------
# Estonian has no gendered pronouns (tema/ta covers he and she), so the five primary seed
# pairs are the kinship/role analogues of the paper's man/woman, boy/girl, male/female
# pairs plus father/mother and brother/sister, exactly as in the Dutch primary set.
SEEDS = {
    'primary':  [('mees', 'naine'), ('poiss', 'tüdruk'), ('mehelik', 'naiselik'),
                 ('isa', 'ema'), ('vend', 'õde')],
    'extended': [('mees', 'naine'), ('poiss', 'tüdruk'), ('mehelik', 'naiselik'),
                 ('isa', 'ema'), ('vend', 'õde'), ('poeg', 'tütar'), ('onu', 'tädi'),
                 ('kuningas', 'kuninganna'), ('härra', 'proua'), ('vanaisa', 'vanaema')],
}
# gendered word pairs never used as seeds: the axis should order male above female in all of them
VALIDATION = [('peigmees', 'pruut'), ('näitleja', 'näitlejanna'), ('laulja', 'lauljatar'),
              ('sõber', 'sõbranna'), ('noormees', 'neiu'), ('isand', 'emand'),
              ('ristiisa', 'ristiema'), ('vanamees', 'vanamoor'), ('abielumees', 'abielunaine'),
              ('meesterahvas', 'naisterahvas'), ('kasuisa', 'kasuema'), ('prints', 'printsess'),
              ('kuningapoeg', 'kuningatütar'), ('vanaonu', 'vanatädi'), ('ärimees', 'ärinaine'),
              ('mehed', 'naised'), ('poisid', 'tüdrukud'), ('isad', 'emad'), ('vennad', 'õed'),
              ('pojad', 'tütred')]
# crude 'lexically gendered' filter for the exclusion check (gendered morphemes, kinship/role stems)
GENDERED_RE = re.compile(r'(?:mees$|mehe|naine|naise|poiss|tüdruk|isa$|ema$|vend$|õde$|poeg$|tütar|härra|proua|'
                         r'preili|kuningas|kuninganna|prints|onu$|tädi|vanaema|vanaisa|abikaasa|pruut|peigmees|'
                         r'neiu|noormees|mehelik|naiselik|tar$|nna$|lesk|isand|emand)')
LANGNAME, SCALE = 'Estonian', '0–10'


# ----------------------------------------------------------------------------
# Vector sources
# ----------------------------------------------------------------------------
class FastTextBin:
    """cc.et.300.bin via the fastText package. Words outside the 2M-word vocabulary are not
    scored unless --oov subword is given (fastText can build them from character n-grams)."""
    def __init__(self, path, oov='skip'):
        import fasttext
        self.m = fasttext.load_model(path)
        words, freqs = self.m.get_words(include_freq=True)
        self.vocab = set(words)
        self.freq = dict(zip(words, (int(f) for f in freqs)))
        self.oov = oov
        self.name = os.path.basename(path)
        self.size = len(words)
    def __call__(self, w):
        if w in self.vocab or self.oov == 'subword':
            v = np.asarray(self.m.get_word_vector(w), dtype=np.float32)
            return v if np.linalg.norm(v) > 0 else None
        wl = w.lower()
        if wl != w and (wl in self.vocab):
            return np.asarray(self.m.get_word_vector(wl), dtype=np.float32)
        return None
    def logfreq(self, w):
        f = self.freq.get(w, self.freq.get(w.lower()))
        return np.log(f) if f else np.nan


class VecText:
    """word2vec/fastText text vectors (.vec / .vec.gz). Only the rows for the words that are
    needed are kept in memory. The file order is the corpus-frequency order, so the log rank
    of a word is used as its frequency measure."""
    def __init__(self, path, needed):
        needed = set(needed) | {w.lower() for w in needed}
        op = gzip.open if path.endswith('.gz') else open
        self.vec, self.rank, dim, n = {}, {}, None, 0
        with op(path, 'rt', encoding='utf8', errors='ignore') as f:
            for i, line in enumerate(f):
                p = line.rstrip('\n').split(' ')
                if i == 0 and len(p) == 2:
                    n, dim = int(p[0]), int(p[1]); continue
                if dim is None: dim = len(p) - 1
                w = ' '.join(p[:len(p) - dim])          # tolerate the rare multi-token key
                if w in needed and w not in self.vec:
                    self.vec[w] = np.array(p[len(p) - dim:], dtype=np.float32); self.rank[w] = i
        self.name = os.path.basename(path); self.size = n or i
    def __call__(self, w):
        v = self.vec.get(w)
        if v is None: v = self.vec.get(w.lower())
        return v
    def logfreq(self, w):
        r = self.rank.get(w, self.rank.get(w.lower()))
        return -np.log(r) if r else np.nan   # higher = more frequent


# ----------------------------------------------------------------------------
# Core computations (identical to the Dutch pipeline)
# ----------------------------------------------------------------------------
def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

_WARNED = set()

def gender_axis(lookup, pairs):
    diffs, used = [], []
    for m, f in pairs:
        vm, vf = lookup(m), lookup(f)
        if vm is None or vf is None:
            if (m, f) not in _WARNED:
                print(f'    seed pair missing in the vectors: ({m}, {f})', file=sys.stderr); _WARNED.add((m, f))
            continue
        diffs.append(unit(vm) - unit(vf)); used.append((m, f))
    if not diffs: raise RuntimeError('no seed pairs available in the vectors')
    return unit(np.mean(diffs, axis=0)), used

def score_terms(lookup, axis, words):
    g = np.full(len(words), np.nan)
    for i, w in enumerate(words):
        v = lookup(w)
        if v is not None: g[i] = float(unit(v) @ axis)
    return g

def fisher_ci(r, n):
    if n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3); q = stats.norm.ppf(0.975)
    return tuple(np.tanh([z - q * se, z + q * se]))

def zs(x):
    x = np.asarray(x, float); return (x - np.nanmean(x)) / np.nanstd(x)

def describe(df, label):
    d = df.dropna(subset=['gender', 'conc'])
    r, p = stats.pearsonr(d.gender, d.conc); lo, hi = fisher_ci(r, len(d)); rho, _ = stats.spearmanr(d.gender, d.conc)
    return dict(specification=label, r=r, ci_lo=lo, ci_hi=hi, p=p, rho=rho, n=len(d))

def ols_beta(df, controls, label):
    cols = ['conc'] + controls
    d = df.dropna(subset=['gender'] + cols)
    X = sm.add_constant(np.column_stack([zs(d[c]) for c in cols]))
    res = sm.OLS(zs(d.gender), X).fit(cov_type='HC1'); ci = res.conf_int()[1]
    return dict(specification=label, r=res.params[1], ci_lo=ci[0], ci_hi=ci[1], p=res.pvalues[1], rho=np.nan, n=len(d))

def deciles(df):
    d = df.dropna(subset=['gender', 'conc']).copy()
    d['decile'] = pd.qcut(d.gender.rank(method='first'), 10, labels=range(1, 11)).astype(int)
    return d.groupby('decile').conc.agg(['mean', 'count']).reset_index()

def validation_accuracy(lookup, axis, used):
    hits = []
    for m, f in VALIDATION:
        if (m, f) in used: continue
        vm, vf = lookup(m), lookup(f)
        if vm is None or vf is None: continue
        hits.append(float(unit(vm) @ axis) > float(unit(vf) @ axis))
    return (np.mean(hits) if hits else np.nan), len(hits)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--norms', required=True, help='data.csv from the Proos & Aigro OSF archive')
    ap.add_argument('--model', default=None, help='path to cc.et.300.bin (fastText binary)')
    ap.add_argument('--vec', default=None, help='path to cc.et.300.vec[.gz] (text vectors)')
    ap.add_argument('--oov', default='skip', choices=['skip', 'subword'],
                    help='words outside the fastText vocabulary: skip (default) or build from subwords (.bin only)')
    ap.add_argument('--min-ratings', type=int, default=1, help='keep words with at least this many concreteness ratings')
    ap.add_argument('--out', default='results_et')
    ap.add_argument('--model-label', default=None, help='legend label for the vectors (default: fastText Estonian vectors for cc.et.*, else file name)')
    ap.add_argument('--no-figure', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- norms --------------------------------------------------------------
    norms = pd.read_csv(a.norms, keep_default_na=False, na_values=[''], usecols=lambda c: c in
                        {'lemma', 'concreteness_index', 'concreteness_sd', 'concreteness_ratingcount'})
    norms = norms.rename(columns={'lemma': 'word', 'concreteness_index': 'conc', 'concreteness_sd': 'conc_sd',
                                  'concreteness_ratingcount': 'n_ratings'})
    for c in ('conc', 'conc_sd', 'n_ratings'): norms[c] = pd.to_numeric(norms[c], errors='coerce')
    norms['word'] = norms.word.astype(str).str.strip()
    norms = norms[(norms.word != '') & norms.conc.notna() & (norms.n_ratings.fillna(0) >= a.min_ratings)]
    norms = norms.groupby('word', as_index=False).agg(conc=('conc', 'mean'), conc_sd=('conc_sd', 'mean'), n_ratings=('n_ratings', 'sum'))
    norms['length'] = norms.word.str.len()
    print(f'[{LANGNAME}] {len(norms):,} rated words; concreteness mean {norms.conc.mean():.2f} (sd {norms.conc.std():.2f}) on a {SCALE} scale')

    # ---- vectors ------------------------------------------------------------
    all_seed_words = {w for s in SEEDS.values() for p in s for w in p} | {w for p in VALIDATION for w in p}
    if a.vec:
        lookup = VecText(a.vec, set(norms.word) | all_seed_words)
    else:
        path = a.model
        if path is None:
            import fasttext.util
            print('Downloading fastText Estonian vectors (cc.et.300.bin.gz, ~4 GB) via fasttext.util ...')
            path = fasttext.util.download_model('et', if_exists='ignore')   # -> 'cc.et.300.bin' in cwd
        lookup = FastTextBin(path, oov=a.oov)
    print(f'  vectors: {lookup.name} ({lookup.size:,} words)')
    norms['logfreq'] = [lookup.logfreq(w) for w in norms.word]

    # ---- main analysis for each seed set --------------------------------------
    rows, dec, seeds_used, scores = [], [], {}, norms[['word', 'conc']].copy()
    for pset, pairs in SEEDS.items():
        axis, used = gender_axis(lookup, pairs); seeds_used[pset] = used
        g = score_terms(lookup, axis, norms.word.tolist())
        df = norms.copy(); df['gender'] = g; scores[f'gender::{pset}'] = g
        acc, nval = validation_accuracy(lookup, axis, used)
        print(f'  seeds={pset}: {len(used)}/{len(pairs)} pairs; {int(np.isfinite(g).sum()):,}/{len(df):,} words scored; '
              f'held-out gendered pairs ordered correctly: {acc:.2f} (n={nval})')
        R = [describe(df, 'all terms (no controls)')]
        seedset = {w for p in used for w in p}
        R.append(describe(df[~(df.word.str.contains(GENDERED_RE) | df.word.isin(seedset))], 'excl. lexically gendered terms + seeds'))
        for q in (0.05, 0.10):
            thr = df.gender.abs().quantile(1 - q); R.append(describe(df[df.gender.abs() < thr], f'excl. {int(q*100)}% most gendered terms'))
        if df.conc_sd.notna().sum() > 100:
            R.append(describe(df[df.conc_sd <= df.conc_sd.median()], 'rater s.d. of concreteness <= median (reliably rated)'))
        R.append(ols_beta(df, [], 'OLS: no controls'))
        if df.logfreq.notna().sum() > 100:
            R.append(ols_beta(df, ['logfreq'], 'OLS: + log frequency'))
        R.append(ols_beta(df, ['length'], 'OLS: + length'))
        if df.logfreq.notna().sum() > 100:
            R.append(ols_beta(df, ['logfreq', 'length'], 'OLS: + frequency + length'))
            dq = df.dropna(subset=['logfreq', 'gender']).copy()
            dq['fq'] = pd.qcut(dq.logfreq.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            for k in range(1, 6):
                R.append(describe(dq[dq.fq == k], f'frequency quintile {k}' + (' (least frequent)' if k == 1 else ' (most frequent)' if k == 5 else '')))
        for r_ in R: r_.update(seeds=pset, validation_accuracy=acc, validation_pairs=nval)
        rows += R
        dd = deciles(df); dd['seeds'] = pset; dec.append(dd)

    # ---- pole sensitivity ------------------------------------------------------
    prim, ext = SEEDS['primary'], SEEDS['extended']
    specs = {'primary set (5 pairs)': prim, 'extended set (10 pairs)': ext}
    for p in prim: specs[f'primary minus {p[0]}/{p[1]}'] = [q for q in prim if q != p]
    for p in ext: specs[f'single pair {p[0]}/{p[1]}'] = [p]
    PS = []
    for label, pairs in specs.items():
        try: axis, used = gender_axis(lookup, pairs)
        except RuntimeError: continue
        g = score_terms(lookup, axis, norms.word.tolist()); ok = np.isfinite(g)
        r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
        d = pd.DataFrame({'w': norms.word, 'g': g}).dropna().sort_values('g')
        PS.append(dict(specification=label, n_pairs=len(used), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()),
                       most_female=', '.join(d.w.head(8)), most_male=', '.join(d.w.tail(8)[::-1])))
    D = np.vstack([unit(unit(lookup(m)) - unit(lookup(f))) for m, f in ext if lookup(m) is not None and lookup(f) is not None])
    U, s, Vt = np.linalg.svd(D, full_matrices=False); pc1 = Vt[0]; pc1 = pc1 if D.mean(0) @ pc1 > 0 else -pc1
    g = score_terms(lookup, unit(pc1), norms.word.tolist()); ok = np.isfinite(g)
    r, p = stats.pearsonr(g[ok], norms.conc.values[ok]); lo, hi = fisher_ci(r, ok.sum())
    d = pd.DataFrame({'w': norms.word, 'g': g}).dropna().sort_values('g')
    PS.append(dict(specification=f'PC1 of extended set (var. expl. {s[0]**2/(s**2).sum():.2f}; loadings {np.round(D@pc1,2).tolist()})',
                   n_pairs=len(D), r=r, ci_lo=lo, ci_hi=hi, p=p, n=int(ok.sum()),
                   most_female=', '.join(d.w.head(8)), most_male=', '.join(d.w.tail(8)[::-1])))

    # ---- write ----------------------------------------------------------------
    S = pd.DataFrame(rows)[['seeds', 'specification', 'r', 'ci_lo', 'ci_hi', 'p', 'rho', 'n', 'validation_accuracy', 'validation_pairs']]
    S.insert(0, 'model', lookup.name); S.insert(0, 'language', LANGNAME)
    S.to_csv(os.path.join(a.out, 'stats_et.csv'), index=False)
    pd.concat(dec).to_csv(os.path.join(a.out, 'deciles_et.csv'), index=False)
    pd.DataFrame(PS).to_csv(os.path.join(a.out, 'pole_sensitivity_et.csv'), index=False)
    scores.to_csv(os.path.join(a.out, 'scores_et.csv'), index=False)
    json.dump(seeds_used, open(os.path.join(a.out, 'seeds_et.json'), 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200); pd.set_option('display.max_colwidth', 60)
    print('\n' + S.drop(columns=['language', 'model']).round(4).to_string(index=False))
    print('\nMean concreteness by gender-score decile (1 = most feminine, 10 = most masculine):')
    print(pd.concat(dec).pivot_table(index='decile', columns='seeds', values='mean').round(2).to_string())
    print('\nSensitivity to seed pairs:')
    print(pd.DataFrame(PS)[['specification', 'n_pairs', 'r', 'ci_lo', 'ci_hi', 'n', 'most_female', 'most_male']].round(3).to_string(index=False))

    # ---- figure (same design as the Dutch panel) ------------------------------
    if not a.no_figure:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        DD = pd.concat(dec)
        n_words = int(S[(S.seeds == 'primary') & (S.specification == 'all terms (no controls)')].n.iloc[0])
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, ax = plt.subplots(figsize=(4.4, 4.3))
        for pset, label, col, mk in [('primary', '5 seed pairs (primary)', '#1f77b4', 'o'), ('extended', '10 seed pairs (extended)', '#17becf', 's')]:
            d = DD[DD.seeds == pset].sort_values('decile'); ax.plot(d.decile, d['mean'], marker=mk, ms=4, lw=1.4, color=col, label=label)
        ax.set_xticks(range(1, 11)); ax.set_xlim(0.5, 10.5)
        ax.set_ylabel(f'Mean Concreteness Rating\n({LANGNAME} norms, {SCALE})')
        ax.set_title(f'{LANGNAME}: Human Judgments of Concreteness', fontsize=10); ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.03, 0.03, f'N = {n_words:,} words', ha='left', va='bottom', transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='0.6', lw=0.6))
        mlabel = a.model_label or (f'fastText {LANGNAME} vectors' if 'cc.et' in lookup.name else lookup.name)
        ax.legend(loc='best', fontsize=8, frameon=False, title=f'Gender axis ({mlabel})', title_fontsize=8)
        ax.set_xlabel('Gender Score Decile', labelpad=34)
        ax.annotate('', xy=(0.02, -0.11), xytext=(0.44, -0.11), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.8))
        ax.annotate('', xy=(0.98, -0.11), xytext=(0.56, -0.11), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.8))
        ax.text(0.23, -0.14, 'More Feminine', ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.text(0.77, -0.14, 'More Masculine', ha='center', va='top', transform=ax.transAxes, fontsize=8)
        for ext_ in ('png', 'pdf'):
            fig.savefig(os.path.join(a.out, f'figS_estonian_replication.{ext_}'), dpi=300, bbox_inches='tight')
        print(f'\nfigure written to {os.path.join(a.out, "figS_estonian_replication.png")}')


if __name__ == '__main__':
    main()
