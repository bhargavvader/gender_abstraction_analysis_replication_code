"""Emit the LaTeX bodies of the two Supplementary Tables (specifications; seed-pair sensitivity) from the results CSVs.
usage: python si_tables.py zh results_zh fig:S-chinese        (Chinese words are wrapped in \\zh{...})"""
import sys, re, pandas as pd, numpy as np
lang, rdir, figlabel = sys.argv[1], sys.argv[2], sys.argv[3]
if lang.startswith('zh'):
    W = lambda w: f'\\zh{{{w}}}'
else:
    W = lambda w: f'\\emph{{{w}}}'
S=pd.read_csv(f'{rdir}/stats_{lang}.csv'); P=pd.read_csv(f'{rdir}/pole_sensitivity_{lang}.csv'); D=pd.read_csv(f'{rdir}/deciles_{lang}.csv')
def fz(x): x=round(float(x),3); return f'{0.0 if x==0 else x:.3f}'
def ci(lo,hi): return f'[${fz(lo)}$, ${fz(hi)}$]'
def pv(p):
    if p<2.2e-16: return '$<2.2\\times10^{-16}$'
    if p>=0.001: return f'${p:.2f}$' if p>=0.01 else f'${p:.3f}$'
    m,e=f'{p:.1e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
def n(x): return f'{int(x):,}'.replace(',','{,}')
labels=[('all terms (no controls)',f'No controls (Supplementary Fig.~\\ref{{{figlabel}}})'),
        ('excl. lexically gendered terms + seeds','Excluding lexically gendered terms and seed words'),
        ('excl. 5% most gendered terms','Excluding the 5\\% most strongly gendered terms'),
        ('excl. 10% most gendered terms','Excluding the 10\\% most strongly gendered terms'),
        ('rater s.d. of concreteness <= median (reliably rated)','Rater s.d.\\ of concreteness $\\leq$ median (reliably rated words)'),
        ('OLS: + log frequency','$\\beta$, controlling for word frequency'),
        ('OLS: + log frequency (fastText rank)','$\\beta$, controlling for word frequency'),
        ('OLS: + length','$\\beta$, controlling for length'),
        ('OLS: + frequency + length','$\\beta$, controlling for frequency and length'),
        ('frequency quintile 1 (least frequent)','Frequency quintile 1 (least frequent)'),
        ('frequency quintile 2','Frequency quintile 2'),('frequency quintile 3','Frequency quintile 3'),('frequency quintile 4','Frequency quintile 4'),
        ('frequency quintile 5 (most frequent)','Frequency quintile 5 (most frequent)')]
print('%% ---- Table A ----')
for key,label in labels:
    a=S[(S.seeds=='primary')&(S.specification==key)]; b=S[(S.seeds=='extended')&(S.specification==key)]
    if a.empty or b.empty: continue
    a=a.iloc[0]; b=b.iloc[0]
    print(f'{label} & ${a.r:.3f}$ {ci(a.ci_lo,a.ci_hi)} & {pv(a.p)} & {n(a.n)} & ${b.r:.3f}$ {ci(b.ci_lo,b.ci_hi)} & {pv(b.p)} & {n(b.n)} \\\\')
print('\n%% ---- Table B ----')
pron=set(w for s in ['pronoun'] for _ in [0] for w in [])
def grp(s):
    if 'pronoun' in s or re.search(r'single pair (他|hij|hem|on)/', s): return 'C'
    if s.startswith('single pair'): return 'B'
    return 'A'
P['grp']=P.specification.map(grp)
fam={'A':'A. Axes built from several kinship/role pairs','B':'B. Single kinship/role pairs','C':'C. Pronoun-based seeds'}
for g in ['A','B','C']:
    if (P.grp==g).sum()==0: continue
    print(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{fam[g]}}}}} \\\\')
    for _,r in P[P.grp==g].iterrows():
        lab=r.specification.replace('primary set (5 pairs)','Primary set (5 pairs)').replace('extended set (10 pairs)','Extended set (10 pairs)').replace('pronoun set','Pronoun set')
        lab=re.sub(r'^primary minus (\S+)/(\S+)',lambda m: f'Primary set without {W(m.group(1))}/{W(m.group(2))}',lab)
        lab=re.sub(r'^single pair (\S+)/(\S+)',lambda m: f'{W(m.group(1))}/{W(m.group(2))}',lab)
        lab=re.sub(r'^PC1 of extended set \(var\. expl\. ([0-9.]+); loadings.*\)$',r'First principal component of the 10 pair differences (\1 of variance)',lab)
        mf=', '.join(W(w) for w in r.most_female.split(', ')[:5]); mm=', '.join(W(w) for w in r.most_male.split(', ')[:5])
        print(f'{lab} & {int(r.n_pairs)} & ${r.r:.3f}$ {ci(r.ci_lo,r.ci_hi)} & {pv(r.p)} & {mf} & {mm} \\\\')
print('\n%% deciles:'); print(D.pivot_table(index='decile',columns='seeds',values='mean').round(2).to_string().replace('\n','\n%% '))
