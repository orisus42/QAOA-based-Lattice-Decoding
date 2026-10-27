"""Frozen-method fresh-cohort results; internal research figures, not poster prose."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = {'full_fixed': 'Full QAOA', 'weak75_frozen': '75% / same angles',
          'weak90_retrained': '90% / trained angles', 'weak75_retrained': '75% / trained angles'}
COLORS = {'full_fixed': '#42576d', 'weak75_frozen': '#00877f',
          'weak90_retrained': '#d17a20', 'weak75_retrained': '#9064a5'}


def main():
    run = Path(sys.argv[1]).resolve()
    s = json.loads((run/'summary.json').read_text())
    cfg = json.loads((run/'config.json').read_text())
    rows = json.loads((run/'per_instance.json').read_text())
    assert s['experiment_id'] == 'EXP-007'
    out = run/'figures'
    suffix = 1
    while out.exists():
        suffix += 1
        out = run/f'figures-v{suffix}'
    out.mkdir()
    shutil.copy2(__file__, out/'plot_source.py')
    sns.set_theme(style='whitegrid', font='DejaVu Sans', font_scale=1.03)
    plt.rcParams.update({'svg.fonttype': 'none', 'axes.edgecolor': '#d5dddc', 'grid.color': '#e5e9e7',
                         'text.color': '#173042', 'axes.labelcolor': '#173042'})
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#fbfaf7')
    fig.subplots_adjust(left=.08, right=.96, top=.79, bottom=.16, hspace=.55, wspace=.28)
    fig.text(.08, .955, 'MODULE-LWE  /  FROZEN METHODS, FRESH LATTICES', fontsize=12, weight='bold', color='#00877f')
    fig.text(.08, .900, 'Does the pruning trade-off carry over?', fontsize=27, weight='bold')
    fig.text(.08, .851, '512 independent examples  /  16 qubits  /  one layer  /  up to 16,384 ideal shots', fontsize=13, color='#536877')
    budgets = [b for b in cfg['budgets'] if b >= 256]
    for name, label in LABELS.items():
        m = s['methods'][name]
        color = COLORS[name]
        a = [100*m['by_budget'][str(b)]['hybrid_success']['mean'] for b in budgets]
        sns.lineplot(x=budgets, y=a, marker='o', color=color, label=label, linewidth=2.2, ax=axs[0, 0])
    axs[0, 0].axhline(100*s['babai']['success']['mean'], color='#a2abad', ls='--', lw=1.5, label='Babai alone')
    axs[0, 0].set(xscale='log', xlabel='Shots per example (log scale)', ylabel='Babai + QAOA exact success (%)')
    axs[0, 0].set_title('A   Solution quality across shot budgets', loc='left', weight='bold', fontsize=12, pad=15)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].get_legend().remove()
    for name in list(LABELS)[1:]:
        vals = s['comparisons_to_full'][name]['by_budget']
        mean = [100*vals[str(b)]['extra_distance_improvement_retention']['estimate'] for b in budgets]
        lo = [100*vals[str(b)]['extra_distance_improvement_retention']['ci95'][0] for b in budgets]
        hi = [100*vals[str(b)]['extra_distance_improvement_retention']['ci95'][1] for b in budgets]
        axs[0, 1].fill_between(budgets, lo, hi, color=COLORS[name], alpha=.10)
        sns.lineplot(x=budgets, y=mean, marker='o', color=COLORS[name], linewidth=2.2, ax=axs[0, 1])
    axs[0, 1].axhline(100, color='#42576d', ls='--', lw=1)
    axs[0, 1].axhline(95, color='#a2abad', ls=':', lw=1)
    axs[0, 1].set(xscale='log', xlabel='Shots per example (log scale)', ylabel='Full-QAOA extra distance benefit retained (%)')
    axs[0, 1].set_title('B   Benefit beyond Babai, with 95% bootstrap intervals', loc='left', weight='bold', fontsize=11, pad=15)
    b = str(cfg['primary_budget'])
    for name in LABELS:
        m = s['methods'][name]
        x = m['compiled_cx']['mean']
        y = 100*m['by_budget'][b]['hybrid_success']['mean']
        axs[1, 0].scatter(x, y, color=COLORS[name], s=130, zorder=3, edgecolor='white', lw=1.5)
        short = {'full_fixed': 'Full', 'weak75_frozen': '75% same', 'weak90_retrained': '90% trained', 'weak75_retrained': '75% trained'}[name]
        axs[1, 0].annotate(short, (x, y), xytext=(0, 14 if name != 'weak75_frozen' else -22), textcoords='offset points', ha='center', fontsize=10)
    axs[1, 0].axhline(100*s['babai']['success']['mean'], color='#a2abad', ls='--', lw=1)
    axs[1, 0].set(xlabel='Mean compiled CX gates per circuit', ylabel='Babai + QAOA exact success (%)', xlim=(0, 245))
    axs[1, 0].margins(y=.28)
    axs[1, 0].set_title('C   Circuit cost versus quality at 1,024 shots', loc='left', weight='bold', fontsize=12, pad=15)
    ratios = []
    cats = []
    for name in list(LABELS)[1:]:
        for row in rows:
            full = row['methods']['full_fixed']['p_optimum']
            candidate = row['methods'][name]['p_optimum']
            if full > 0 and candidate > 0:
                ratios.append(np.log2(candidate/full))
                cats.append(LABELS[name])
    palette = [COLORS[n] for n in list(LABELS)[1:]]
    sns.violinplot(x=cats, y=ratios, hue=cats, palette=palette, inner='quart', cut=0,
                   linewidth=1, legend=False, ax=axs[1, 1])
    axs[1, 1].axhline(0, color='#42576d', ls='--', lw=1)
    axs[1, 1].set(ylabel='Single-shot success ratio vs full QAOA (log2)', xlabel='')
    axs[1, 1].set_xticks(range(3), ['75%\nsame angles', '90%\ntrained angles', '75%\ntrained angles'])
    axs[1, 1].set_title('D   Individual gains and losses remain visible', loc='left', weight='bold', fontsize=12, pad=15)
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.071, .092), ncol=5, frameon=False, fontsize=10)
    fig.text(.08, .068, 'All curves use exact ideal expectations. Shading in B resamples paired independent lattice groups; it is not hardware noise.', fontsize=10, color='#536877')
    fig.text(.08, .040, f'Fixed tiny parameters n=2, k=2, q=17. Box covers a global optimum in {s["box_covers_optimum_count"]}/512 cases. No fresh-data tuning or filtering.', fontsize=10, color='#536877')
    fig.text(.08, .013, 'All-to-all compilation; no device routing or QPU use. Strong classical controls are recorded in the notebook and raw results.', fontsize=10, color='#536877')
    for ext in ['svg', 'png']:
        fig.savefig(out/f'fresh_pruning.{ext}', dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest = json.loads((run/'manifest.json').read_text())
    for path in out.iterdir():
        manifest['output_sha256'][path.relative_to(run).as_posix()] = sha256(path.read_bytes()).hexdigest()
    manifest['latest_figure'] = (out/'fresh_pruning.png').relative_to(run).as_posix()
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
