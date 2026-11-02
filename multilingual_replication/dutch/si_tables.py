"""Emit the LaTeX bodies of the two Dutch Supplementary Tables (specifications; seed-pair sensitivity) from results_nl/.
usage: python si_tables.py            (reads results_nl/stats_nl.csv and results_nl/pole_sensitivity_nl.csv)"""
import re, pandas as pd
S = pd.read_csv('results_nl/stats_nl.csv'); P = pd.read_csv('results_nl/pole_sensitivity_nl.csv')

def fz(x): x = round(float(x), 3); return f'{0.0 if x == 0 else x:.3f}'
def ci(lo, hi): return f'[${fz(lo)}$, ${fz(hi)}$]'
def pv(p):
    if p < 2.2e-16: return '$<2.2\\times10^{-16}$'
    if p >= 0.001: return f'${p:.2f}$' if p >= 0.01 else f'${p:.3f}$'
    m, e = f'{p:.1e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
def n(x): return f'{int(x):,}'.replace(',', '{,}')

labels = [('all terms (no controls)', 'No controls (Supplementary Fig.~\\ref{fig:S-dutch})'),
          ('excl. lexically gendered terms + seeds', 'Excluding lexically gendered terms and seed words'),
          ('excl. 5% most gendered terms', 'Excluding the 5\\% most strongly gendered terms'),
          ('excl. 10% most gendered terms', 'Excluding the 10\\% most strongly gendered terms'),
          ('OLS: + log frequency', '$\\beta$, controlling for word frequency'),
          ('OLS: + length', '$\\beta$, controlling for length'),
          ('OLS: + frequency + length', '$\\beta$, controlling for frequency and length'),
          ('OLS: + frequency + length + AoA', '$\\beta$, controlling for frequency, length and age of acquisition'),
          ('frequency quintile 1 (least frequent)', 'Frequency quintile 1 (least frequent)'),
          ('frequency quintile 2', 'Frequency quintile 2'), ('frequency quintile 3', 'Frequency quintile 3'),
          ('frequency quintile 4', 'Frequency quintile 4'), ('frequency quintile 5 (most frequent)', 'Frequency quintile 5 (most frequent)')]
print('%% ---- Table A: specifications (primary | extended) ----')
for key, label in labels:
    a = S[(S.seeds == 'primary') & (S.specification == key)].iloc[0]; b = S[(S.seeds == 'extended') & (S.specification == key)].iloc[0]
    print(f'{label} & ${a.r:.3f}$ {ci(a.ci_lo, a.ci_hi)} & {pv(a.p)} & {n(a.n)} & ${b.r:.3f}$ {ci(b.ci_lo, b.ci_hi)} & {pv(b.p)} & {n(b.n)} \\\\')

print('\n%% ---- Table B: seed-pair sensitivity ----')
def grp(s):
    if 'pronoun' in s or re.search(r'single pair (hij|hem)/', s): return 'C'
    if s.startswith('single pair'): return 'B'
    return 'A'
P['grp'] = P.specification.map(grp)
fam = {'A': 'A. Axes built from several unambiguous kinship/role pairs', 'B': 'B. Single kinship/role pairs',
       'C': 'C. Pronoun-based seeds (\\emph{zij} = \\emph{they}; \\emph{haar} = \\emph{hair})'}
for g in ['A', 'B', 'C']:
    print(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{fam[g]}}}}} \\\\')
    for _, r in P[P.grp == g].iterrows():
        lab = r.specification.replace('primary set (5 pairs)', 'Primary set (5 pairs)').replace('extended set (10 pairs)', 'Extended set (10 pairs)')
        lab = re.sub(r'^primary minus (\S+)/(\S+)', r'Primary set without \\emph{\1}/\\emph{\2}', lab)
        lab = re.sub(r'^single pair (\S+)/(\S+)', r'\\emph{\1}/\\emph{\2}', lab)
        lab = re.sub(r'^PC1 of extended set \(var\. expl\. ([0-9.]+);.*\)$', r'First principal component of the 10 pair differences (\1 of variance)', lab)
        lab = lab.replace('pronoun set', 'Pronoun set (\\emph{man}/\\emph{vrouw}, \\emph{hij}/\\emph{zij}, \\emph{hem}/\\emph{haar}, \\emph{mannelijk}/\\emph{vrouwelijk}, \\emph{jongen}/\\emph{meisje})')
        mf = ', '.join(f'\\emph{{{w}}}' for w in r.most_female.split(', ')[:5]); mm = ', '.join(f'\\emph{{{w}}}' for w in r.most_male.split(', ')[:5])
        print(f'{lab} & {int(r.n_pairs)} & ${r.r:.3f}$ {ci(r.ci_lo, r.ci_hi)} & {pv(r.p)} & {mf} & {mm} \\\\')
