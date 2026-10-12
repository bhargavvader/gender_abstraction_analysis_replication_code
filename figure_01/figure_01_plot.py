# ==============================================================================
# figure_01_plot.py  (= fig1_plot_12models.py, final version of 5 Oct 2026)
#
# Renders fig1_text_data.png / .pdf from the decile means and inset Ns written
# by figure_01_compute.py. This is the script that produced the Fig. 1 file in
# the manuscript; figure_01.R is the ggplot2/patchwork equivalent for the
# local R pipeline (same registry, same geometry).
#
#   A  Human Judgments of Concreteness          (conc_brysbaert;       words + phrases)
#   B  AI Judgments of Concreteness             (conc_gpt_prob;        phrases only)
#   C  Human Judgments of Sensorimotor Strength (lanc_mean_perceptual; words only)
#
# Conventions: scales::hue_pal() colours (one per model, identical in every
# panel), N inset top-right, direction arrows ("More Feminine" / "More
# Masculine") under each panel, shared bottom legend (3 rows x 4 columns, font
# auto-shrunk until it fits), panels A and B share an identical y-axis range
# (both are 1-5 concreteness ratings); panel C keeps its own range.
# ==============================================================================
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as c
from paths import results
OUT = results('figure_01')

# ---- scales::hue_pal() replica: evenly spaced hues in HCL (c=100, l=65), D65 white point ----
def hcl_to_hex(h, C=100, L=65):
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    un = 4*Xn/(Xn+15*Yn+3*Zn); vn = 9*Yn/(Xn+15*Yn+3*Zn)
    u, v = C*np.cos(np.radians(h)), C*np.sin(np.radians(h))
    Y = Yn*((L+16)/116)**3 if L > 8 else Yn*L/903.3
    up, vp = u/(13*L)+un, v/(13*L)+vn
    X = Y*9*up/(4*vp); Z = Y*(12-3*up-20*vp)/(4*vp)
    M = np.array([[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]])
    rgb = M @ (np.array([X,Y,Z])/100)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-0.055)
    return '#%02x%02x%02x' % tuple(int(round(255*x)) for x in rgb)
def hue_pal(n):
    hs = np.linspace(15, 375, n+1)[:n]
    return [hcl_to_hex(h % 360) for h in hs]

dec = pd.read_csv(os.path.join(OUT, 'fig1_text_data_decile_means.csv'))
Ns  = json.load(open(os.path.join(OUT, 'fig1_Ns.json')))
models = [n for n,_ in c.REGISTRY]
pal = dict(zip(models, hue_pal(len(models))))

FIG_W, FIG_H, DPI = 16, 9.2, 400
plt.rcParams.update({'font.family':'DejaVu Sans', 'axes.linewidth':1.0})
GREY20 = '#333333'
fig = plt.figure(figsize=(FIG_W, FIG_H))
# panel geometry (figure fractions): leave room for 3-row legend at bottom and direction key
left, right, bottom, top = 0.072, 0.988, 0.335, 0.905
gap = 0.098
pw = (right-left-2*gap)/3
axes = []
# panels A and B are both concreteness ratings (1-5): give them an identical y-axis range
ab = dec[dec.panel.isin(['A', 'B'])].value; pad = 0.03 * (ab.max() - ab.min()); AB_YLIM = (ab.min() - pad, ab.max() + pad)
for i, tag in enumerate('ABC'):
    ax = fig.add_axes([left+i*(pw+gap), bottom, pw, top-bottom]); axes.append(ax)
    p = c.PANELS[tag]
    d = dec[dec.panel==tag]
    for mname in models:
        s = d[d.model==mname].sort_values('decile')
        if s.empty: continue
        ax.plot(s.decile, s.value, '-o', color=pal[mname], lw=2.2, ms=6.5, mec=pal[mname], zorder=3)
    ax.set_title(p['title'], fontsize=21, pad=14, linespacing=1.1)
    ax.set_ylabel(p['ylab'], fontsize=20, linespacing=1.1, labelpad=10)
    ax.set_xticks(range(1,11)); ax.tick_params(axis='both', labelsize=18, colors=GREY20, length=4)
    for sp in ax.spines.values(): sp.set_color('black'); sp.set_linewidth(1.0)
    ax.grid(True, which='major', color='#ebebeb', lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0.55, 10.45)
    if tag in ('A', 'B'): ax.set_ylim(*AB_YLIM)
    # N inset (top right)
    unit = dict(word='words', phrase='phrases')
    label = '\n'.join(f'N = {n:,} {unit[k]}' for k,n in Ns[tag].items())
    ax.text(0.975, 0.965, label, transform=ax.transAxes, ha='right', va='top', fontsize=17.5, linespacing=1.15, color='black',
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.15', fc='white', ec='black', lw=0.8), zorder=5)
    # panel tag
    ax.text(-0.235, 1.085, tag, transform=ax.transAxes, fontsize=32, fontweight='bold', ha='left', va='bottom')
    # x title + direction key (arrows on one row, labels beneath)
    ax.set_xlabel('Gender Score Decile', fontsize=22, labelpad=62)
    y_arrow, y_text = -0.105, -0.165
    arr = dict(arrowstyle='-|>,head_length=0.7,head_width=0.35', color=GREY20, lw=2.4, shrinkA=0, shrinkB=0)
    ax.annotate('', xy=(0.02, y_arrow), xytext=(0.45, y_arrow), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arr, annotation_clip=False)
    ax.annotate('', xy=(0.98, y_arrow), xytext=(0.55, y_arrow), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arr, annotation_clip=False)
    ax.text(0.235, y_text, 'More Feminine',  transform=ax.transAxes, ha='center', va='center', fontsize=16, color=GREY20)
    ax.text(0.765, y_text, 'More Masculine', transform=ax.transAxes, ha='center', va='center', fontsize=16, color=GREY20)

# shared legend: 12 models, 3 rows, filled by row (4 columns)
NROW = 3; NCOL = int(np.ceil(len(models)/NROW))
# matplotlib fills legends column-major; reorder so the visual order is row-major (ggplot byrow = TRUE)
order = [r + c_*NROW for c_ in range(NCOL) for r in range(NROW)]
grid = [None]*(NCOL*NROW)
for i, m in enumerate(models): grid[(i % NCOL)*NROW + i//NCOL] = m
cols = [g for g in grid if g is not None]
handles = [Line2D([0],[0], color=pal[m], marker='o', ms=10, lw=3.0, mec=pal[m]) for m in cols]
# shrink the legend font until all columns fit inside the figure width
fig.canvas.draw()
for fs in [16, 15.5, 15, 14.5, 14, 13.5, 13]:
    leg = fig.legend(handles, cols, loc='lower center', ncol=NCOL, frameon=False, fontsize=fs, handlelength=2.0, handletextpad=0.55,
                     columnspacing=1.0, labelspacing=0.5, bbox_to_anchor=(0.5, 0.008))
    fig.canvas.draw()
    w = leg.get_window_extent().width / (FIG_W*fig.dpi)
    if w <= 0.985: break
    leg.remove()
print('legend font', fs, 'width frac', round(w,3))
for t in leg.get_texts(): t.set_color('black')
fig.savefig(os.path.join(OUT, 'fig1_text_data.png'), dpi=DPI); fig.savefig(os.path.join(OUT, 'fig1_text_data.pdf'))
print('saved', os.path.join(OUT, 'fig1_text_data.png'))
