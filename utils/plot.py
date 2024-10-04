"""Utility functions for plotting"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# add latex
import os
os.environ["PATH"] += os.pathsep + os.path.expanduser('~') + '/texlive/2023/bin/x86_64-linux'

# graphing
plt.rcParams.update({
    'font.size': 13,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{libertine}\usepackage[libertine]{newtxmath} \usepackage{sfmath}',
    'font.family': 'sans-serif',
})
palette = plt.cm.jet(np.linspace(0,1,21))
all_colors = [matplotlib.colors.to_hex(color, keep_alpha=True) for color in palette]

markers = ['s', '*', 'o', '^', 'p', '1', 'P', 'X']
colors = [all_colors[-1], all_colors[5],  all_colors[-3], all_colors[7], all_colors[15]]

def plot_barv(ax, resultss, labels, upper_bounds=True, title=None, xticklabels=None):
    """Plot vertical bars"""
    if xticklabels is None:
        theor_epses = [eps for eps in resultss[0]['theor_eps'].unique()]
        xticklabels = [str(eps) for eps in theor_epses]
        ax.set_xlabel('Theoretical $\\varepsilon$')
    xticks = np.arange(len(xticklabels))
        
    if len(resultss) == 1:
        width = 0.4
        curr_colors = [colors[4]]
    elif len(resultss) == 2:
        width = 0.4
        curr_colors = [colors[4], colors[1]]
    elif len(resultss) == 3:
        width = 0.25
        curr_colors = [colors[4], colors[1], all_colors[-1]]
    elif len(resultss) == 4:
        width = 0.2
        curr_colors = [colors[4], colors[1], all_colors[-1], colors[3]]
    elif len(resultss) == 5:
        width = 0.15
        curr_colors = [all_colors[15], all_colors[5],  all_colors[-3], all_colors[7], all_colors[-1]]
    elif len(resultss) == 6:
        width = 0.125
        curr_colors = [all_colors[15], all_colors[5],  all_colors[-3], all_colors[7], all_colors[-1], all_colors[16]]
    
    if len(resultss) % 2 == 0:
        # even
        seed_pos = [(i + 0.5) * width for i in range(len(resultss) // 2)]
        positions = [-pos for pos in reversed(seed_pos)] + seed_pos
    else:
        # odd
        seed_pos = [i * width for i in range(len(resultss) // 2 + 1)]
        positions = [-pos for pos in reversed(seed_pos[1:])] + seed_pos

    offset = 0.5

    for pos, results, color, label in zip(positions, resultss, curr_colors, labels):
        yerr = results['emp_eps_std'] if 'emp_eps_std' in results else None
        ax.bar(xticks + pos, results['emp_eps_mean'] + offset, width=width, bottom=-offset, zorder=2, color=color, label=label, yerr=yerr, capsize=3)

    ax.set_xticks(xticks)
    ax.set_xticklabels([label.replace(' ', '\n') for label in xticklabels])

    ax.set_ylim(-offset, 10+offset)
    ax.set_yticks(np.arange(11))
    ax.set_ylabel('Empirical $\\varepsilon_{emp}$')
    ax.grid(color='#DCDCDC', linestyle='-', linewidth=1, zorder=0)

    if upper_bounds:
        for i, eps in enumerate(theor_epses):
            ax.plot([i - 0.45, i + 0.45], [eps, eps], color='red', linestyle='--', label='Theoretical $\\varepsilon$' if i == 0 else None)

    if title is not None:
        ax.set_title(title)