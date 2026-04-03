#!/usr/bin/env python3
"""Generate the manuscript figures from the released data tables."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch, Wedge

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

STROKE_BLACK = [pe.withStroke(linewidth=1.6, foreground='black')]
STROKE_WHITE = [pe.withStroke(linewidth=1.4, foreground='white')]
PI_TICKS = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
PI_TICKLABELS = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']


def data_file(name: str) -> Path:
    p = DATA / name
    if p.exists():
        return p
    raise FileNotFoundError(name)


def load_results_summary() -> dict:
    p = DATA / 'manuscript_summary.json'
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_additional_datasets() -> None:
    needed = [
        DATA / 'bbh_disorder_scan.csv',
        DATA / 'bhz_counterrotating_loop_quality.csv',
        DATA / 'bhz_width_scaling.csv',
        DATA / 'bhz_top_loop_phases.csv',
        DATA / 'bhz_corotating_loop_phases.csv',
        DATA / 'bhz_counterrotating_loop_phases.csv',
    ]
    if all(p.exists() for p in needed):
        return
    subprocess.run([sys.executable, str(ROOT / 'compute_auxiliary_datasets.py')], check=True)


def pivot(df: pd.DataFrame, index: str, columns: str, values: str):
    x = np.array(sorted(df[index].unique()))
    y = np.array(sorted(df[columns].unique()))
    arr = df.pivot(index=index, columns=columns, values=values).reindex(index=x, columns=y).to_numpy()
    return x, y, arr


def savefig(fig: plt.Figure, stem: str):
    fig.savefig(FIG / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def add_panel_labels(fig: plt.Figure, axes, labels, dx=0.02, dy=0.01):
    fig.canvas.draw()
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        x = max(0.001, bb.x0 - dx)
        y = min(0.995, bb.y1 + dy)
        fig.text(x, y, lab, fontsize=10.5, fontweight='bold', ha='left', va='bottom')


def add_cbar(fig: plt.Figure, ax: plt.Axes, im, width=0.012, pad=0.006, ticks=None, ticklabels=None, label: str | None = None):
    bb = ax.get_position()
    cax = fig.add_axes([bb.x1 + pad, bb.y0, width, bb.height])
    cb = fig.colorbar(im, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)
    if label is not None:
        cb.set_label(label, rotation=90, labelpad=4)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.6)
    return cb


def sci_math(x: float, digits: int = 1) -> str:
    if x == 0 or abs(x) < 1e-18:
        return r'$0$'
    exp = int(math.floor(math.log10(abs(x))))
    coeff = x / (10 ** exp)
    coeff_str = f'{coeff:.{digits}f}'.rstrip('0').rstrip('.')
    if coeff_str == '1':
        return rf'$10^{{{exp}}}$'
    return rf'${coeff_str}\times10^{{{exp}}}$'


def signed_sci_math(x: float, digits: int = 1) -> str:
    if x == 0 or abs(x) < 1e-18:
        return r'$0$'
    sign = '-' if x < 0 else ''
    x = abs(x)
    exp = int(math.floor(math.log10(x)))
    coeff = x / (10 ** exp)
    coeff_str = f'{coeff:.{digits}f}'.rstrip('0').rstrip('.')
    if coeff_str == '1':
        return rf'${sign}10^{{{exp}}}$'
    return rf'${sign}{coeff_str}\times10^{{{exp}}}$'


def set_sci_colorbar(cb, ticks, digits: int = 1):
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([sci_math(t, digits=digits) for t in ticks])


def set_sci_axis_ticks(ax: plt.Axes, ticks, axis: str = 'y', digits: int = 1, color: str | None = None):
    labels = [sci_math(t, digits=digits) for t in ticks]
    if axis == 'y':
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        if color is not None:
            ax.tick_params(axis='y', labelcolor=color)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if color is not None:
            ax.tick_params(axis='x', labelcolor=color)


def fit_label_math(coeff: float, power: int, intercept: float, var: str = 'B', coeff_digits: int = 2, inter_digits: int = 2) -> str:
    sign = '+' if intercept >= 0 else '-'
    val = abs(intercept)
    if val < 1e-18:
        tail = ''
    else:
        exp = int(math.floor(math.log10(val)))
        mant = val / (10 ** exp)
        mant_str = f'{mant:.{inter_digits}f}'.rstrip('0').rstrip('.')
        tail = rf' {sign} {mant_str}\times 10^{{{exp}}}'
    return rf'${coeff:.{coeff_digits}f}\,{var}^{{{power}}}{tail}$'


def right_label_above(ax: plt.Axes, text: str, *, color: str = 'black', x: float = 1.01, y: float = 1.015, fontsize: float = 7.6):
    ax.text(x, y, text, transform=ax.transAxes, ha='left', va='bottom', color=color, fontsize=fontsize)


def outlined_text(ax: plt.Axes, x: float, y: float, text: str, **kwargs):
    peff = kwargs.pop('path_effects', STROKE_BLACK)
    return ax.text(x, y, text, path_effects=peff, **kwargs)


def set_pi_ticks(ax: plt.Axes, x: bool = True, y: bool = True):
    if x:
        ax.set_xticks(PI_TICKS, PI_TICKLABELS)
    if y:
        ax.set_yticks(PI_TICKS, PI_TICKLABELS)


def periodic_extend_last(arr: np.ndarray, axis: int) -> np.ndarray:
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(0, 1)
    return np.concatenate([arr, arr[tuple(sl)]], axis=axis)


def annulus_heatmap(ax: plt.Axes, x: np.ndarray, y: np.ndarray, arr: np.ndarray, *, cmap: str, vmin=None, vmax=None):
    arr_ext = periodic_extend_last(arr, axis=1)
    im = ax.imshow(
        arr_ext.T,
        origin='lower',
        extent=[float(x.min()), float(x.max()), 0.0, 2.0 * math.pi],
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return im


def nice_linear_ticks(vmin: float, vmax: float, n: int = 4) -> np.ndarray:
    vals = np.linspace(vmin, vmax, n)
    rounded = np.array([float(f'{v:.2g}') for v in vals])
    # preserve order and uniqueness
    uniq = []
    for v in rounded:
        if not uniq or abs(v - uniq[-1]) > 1e-12:
            uniq.append(v)
    return np.array(uniq, dtype=float)


def set_decimal_colorbar(cb, ticks, digits: int = 2):
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f'{t:.{digits}f}'.rstrip('0').rstrip('.') for t in ticks])
    cb.ax.yaxis.get_offset_text().set_visible(False)


def spread_row_axes(axes, shifts, width_shrink: float = 0.94):
    for ax, dx in zip(axes, shifts):
        bb = ax.get_position()
        new_w = bb.width * width_shrink
        new_x0 = bb.x0 + dx + 0.5 * (bb.width - new_w)
        ax.set_position([new_x0, bb.y0, new_w, bb.height])




# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(8.6, 2.2))
    ax.set_axis_off()
    xs = np.linspace(0.09, 0.91, 5)
    labels = [
        'Hamiltonian family\n$H(\\lambda)$',
        'isolated quartet\n$P_4(\\lambda)$',
        'compressed observables\n$\\widetilde O_A,\\widetilde O_B$',
        'local frame\n$F(\\lambda)$',
        'loop holonomy\n$U(\\mathcal{C})$',
    ]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        w, h = 0.15, 0.38
        patch = FancyBboxPatch(
            (x - w / 2, 0.5 - h / 2),
            w,
            h,
            boxstyle='round,pad=0.02,rounding_size=0.02',
            facecolor='0.97',
            edgecolor='0.4',
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x, 0.5, lab, ha='center', va='center', fontsize=9.2)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + 0.09, 0.5), (xs[i + 1] - 0.09, 0.5), arrowstyle='->', mutation_scale=11, lw=1.0, color='0.4'))
    ax.text(0.5, 0.12, 'success criterion: pointwise qubits are stable, but loop transport leaves $U(2)\\otimes U(2)$', ha='center', va='center', fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig(fig, 'fig_pipeline')


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------


def draw_ssh():
    summary = load_results_summary()
    grid = pd.read_csv(data_file('ssh_torus_grid.csv'))
    qual = pd.read_csv(data_file('ssh_left_loop_quality.csv'))
    size = pd.read_csv(data_file('ssh_size_scaling.csv'))
    holo = pd.read_csv(data_file('ssh_left_loop_holonomy_matrix.csv'), index_col=0)
    _, _, joint = pivot(grid, 'theta_L', 'theta_R', 'joint')

    fig = plt.figure(figsize=(7.35, 4.85))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.965, top=0.95, bottom=0.12, wspace=0.62, hspace=0.56)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    spread_row_axes(axs[3:], shifts=[-0.010, 0.000, 0.012], width_shrink=0.92)

    # (a)
    ax = axs[0]
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.08, 0.37), 0.08, 0.26, fc='0.92', ec='black', lw=0.8))
    ax.add_patch(Rectangle((0.84, 0.37), 0.08, 0.26, fc='0.92', ec='black', lw=0.8))
    x0, x1 = 0.16, 0.84
    for i in range(6):
        xx = x0 + (x1 - x0) * (i + 1) / 7
        ax.add_patch(Circle((xx, 0.5), 0.028, fc='white', ec='0.2', lw=0.8))
        if i < 5:
            xx2 = x0 + (x1 - x0) * (i + 2) / 7
            ax.plot([xx + 0.03, xx2 - 0.03], [0.5, 0.5], color='tab:red' if i % 2 == 0 else 'tab:blue', lw=1.1)
    ax.annotate('', xy=(0.10, 0.67), xytext=(0.14, 0.77), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.annotate('', xy=(0.90, 0.67), xytext=(0.86, 0.77), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.text(0.085, 0.80, r'$\theta_L$', ha='left')
    ax.text(0.885, 0.80, r'$\theta_R$', ha='right')
    ax.text(0.5, 0.22, r'local qubits: edge $\otimes$ spin', ha='center')
    ax.text(0.5, 0.10, 'independent edge Zeeman angles', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (b)
    ax = axs[1]
    im = ax.imshow(joint.T, origin='lower', extent=[0, 2, 0, 2], cmap='magma', aspect='equal', interpolation='bilinear')
    ax.plot([0, 2], [0, 2], color='limegreen', lw=1.2)
    outlined_text(ax, 1.21, 1.17, r'$C_{\rm diag}$', fontsize=8, rotation=27, color='white')
    outlined_text(ax, 1.62, 0.10, r'$C_L$', fontsize=8, color='white')
    outlined_text(ax, 0.05, 1.72, r'$C_R$', fontsize=8, rotation=90, va='center', color='white')
    ax.set_xlabel(r'$\theta_L/\pi$')
    ax.set_ylabel(r'$\theta_R/\pi$')
    cb = add_cbar(fig, ax, im)
    set_sci_colorbar(cb, ticks=[2e-4, 5e-4, 1e-3], digits=1)

    # (c)
    ax = axs[2]
    arr = holo.to_numpy()
    im = ax.imshow(arr, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
    labs = [r'$L\!\downarrow$', r'$L\!\uparrow$', r'$R\!\downarrow$', r'$R\!\uparrow$']
    ax.set_xticks(range(4), labels=labs, rotation=35, ha='right')
    ax.set_yticks(range(4), labels=labs)
    for i in range(4):
        for j in range(4):
            val = arr[i, j]
            color = 'white' if val > 0.55 else 'black'
            peff = STROKE_BLACK if color == 'white' else None
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.5, color=color, path_effects=peff)
    ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, ec='0.25', lw=1.0))
    ax.add_patch(Rectangle((1.5, 1.5), 2, 2, fill=False, ec='0.25', lw=1.0))

    # (d)
    ax = axs[3]
    s = qual['s']
    ax.plot(s, qual['deltaA'], lw=1.25, label=r'$\delta_A$')
    ax.plot(s, qual['deltaB'], lw=1.25, label=r'$\delta_B$')
    ax.plot(s, qual['gap'], lw=1.25, label=r'$\Delta_4$')
    ax.set_xlabel(r'loop coordinate $s$ on $C_L$')
    ax.set_ylabel('gap / split scale')
    ax2 = ax.twinx()
    ax2.plot(s, qual['joint'], lw=1.35, ls='--', color='tab:blue', label=r'$\epsilon_{\rm joint}$')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylabel('')
    set_sci_axis_ticks(ax2, [2e-4, 4e-4, 8e-4], axis='y', digits=1, color='tab:blue')
    right_label_above(ax, r'$\epsilon_{\rm joint}$', color='tab:blue')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='lower right', bbox_to_anchor=(0.90, 0.17), ncol=2, frameon=False, fontsize=6.8)
    ax.text(0.03, 0.06, f"min overlap = {qual['frame_overlap'].min():.4f}", transform=ax.transAxes, fontsize=6.2)

    # (e)
    ax = axs[4]
    cats = [r'$C_L$', r'$C_R$', r'$C_{\rm diag}$']
    dvals = [summary['ssh']['loops']['C_L']['D_loc_strict'], summary['ssh']['loops']['C_R']['D_loc_strict'], summary['ssh']['loops']['C_diag']['D_loc_strict']]
    epvals = [summary['ssh']['loops']['C_L']['ep_mean'], summary['ssh']['loops']['C_R']['ep_mean'], summary['ssh']['loops']['C_diag']['ep_mean']]
    xx = np.arange(len(cats))
    ax.plot(xx, dvals, 'o-', lw=1.2, ms=4.5, label=r'$D_{\rm loc}$')
    ax.set_xticks(xx, cats)
    ax.set_ylabel(r'$D_{\rm loc}$')
    ax.set_xlim(-0.4, len(cats) - 0.6)
    ymin = min(dvals) - 0.01
    ymax = max(dvals) + 0.0055
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(dvals):
        if v > ymax - 0.006:
            ytxt = max(v - 0.0017, ymin + 0.001)
            va = 'top'
        else:
            ytxt = min(v + 0.0012, ymax - 0.0015)
            va = 'bottom'
        ax.text(i, ytxt, f'{v:.3f}', ha='center', va=va, fontsize=7.5)
    ax2 = ax.twinx()
    ax2.plot(xx, epvals, 's--', color='tab:orange', lw=1.1, ms=3.8, label=r'$e_p$')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('')
    right_label_above(ax, r'$e_p$', color='tab:orange')
    ax.legend(ax.get_lines() + ax2.get_lines(), [r'$D_{\rm loc}$', r'$e_p$'], loc='upper right', frameon=False)

    # (f)
    ax = axs[5]
    ax.plot(size['N'], size['D_loc'], 'o-', lw=1.2, ms=4.5)
    ax.set_xlabel(r'chain length $N$')
    ax.set_ylabel(r'$D_{\rm loc}(C_L)$')
    ax.set_ylim(size['D_loc'].min() - 1e-5, size['D_loc'].max() + 1e-5)
    nvals = size['N'].astype(int).tolist()
    ax.set_xticks(nvals)
    ax.set_xticklabels([str(n) for n in nvals])

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_ssh')


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------


def draw_bbh():
    summary = load_results_summary()
    grid = pd.read_csv(data_file('bbh_torus_grid.csv'))
    qual = pd.read_csv(data_file('bbh_diagonal_loop_quality.csv'))
    dens = pd.read_csv(data_file('bbh_corner_density_reference.csv'), index_col=0).to_numpy()
    _, _, nloc = pivot(grid, 'theta_x', 'theta_y', 'nloc')

    fig = plt.figure(figsize=(7.35, 4.85))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.965, top=0.95, bottom=0.12, wspace=0.62, hspace=0.56)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    spread_row_axes(axs[3:], shifts=[-0.008, 0.000, 0.014], width_shrink=0.92)

    # (a)
    ax = axs[0]
    ax.set_axis_off()
    for iy in range(5):
        for ix in range(5):
            fc = '#d8e5f0' if (ix, iy) in [(0, 0), (4, 0), (0, 4), (4, 4)] else '0.96'
            ax.add_patch(Rectangle((0.11 + 0.15 * ix, 0.13 + 0.15 * iy), 0.10, 0.10, fc=fc, ec='0.2', lw=0.8))
    ax.annotate('', xy=(0.15, 0.93), xytext=(0.85, 0.93), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.annotate('', xy=(0.88, 0.17), xytext=(0.88, 0.87), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.text(0.50, 0.95, r'$\theta_x$', ha='center', va='bottom')
    ax.text(0.90, 0.52, r'$\theta_y$', rotation=90, va='center')
    ax.text(0.5, 0.03, r'local qubits: $x$-side $\otimes$ $y$-side', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (b)
    ax = axs[1]
    im = ax.imshow(dens, cmap='YlOrRd', origin='upper', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    add_cbar(fig, ax, im)

    # (c)
    ax = axs[2]
    im = ax.imshow(nloc.T, origin='lower', extent=[0, 2, 0, 2], cmap='viridis', aspect='equal', interpolation='bilinear')
    ax.plot([0, 2], [0, 2], color='white', lw=1.25)
    outlined_text(ax, 1.08, 1.15, r'$C_{\rm diag}$', rotation=27, fontsize=8, color='white')
    outlined_text(ax, 1.62, 0.08, r'$C_x$', fontsize=8, color='white')
    outlined_text(ax, 0.05, 1.72, r'$C_y$', rotation=90, va='center', fontsize=8, color='white')
    ax.set_xlabel(r'$\theta_x/\pi$')
    ax.set_ylabel(r'$\theta_y/\pi$')
    add_cbar(fig, ax, im)

    # (d)
    ax = axs[3]
    ax.plot(qual['s'], qual['deltaA'], lw=1.25, label=r'$\delta_A$')
    ax.plot(qual['s'], qual['deltaB'], lw=1.25, label=r'$\delta_B$')
    ax.plot(qual['s'], qual['gap'], lw=1.25, label=r'$\Delta_4$')
    ax.set_xlabel(r'loop coordinate $s$ on $C_{\rm diag}$')
    ax.set_ylabel('gap / split scale')
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.58), frameon=False, ncol=1)

    # (e)
    ax = axs[4]
    ax.plot(qual['s'], qual['joint'], lw=1.35, label=r'$\epsilon_{\rm joint}$')
    ax.set_xlabel(r'loop coordinate $s$ on $C_{\rm diag}$')
    ax.set_ylabel(r'$\epsilon_{\rm joint}$')
    ax2 = ax.twinx()
    ax2.plot(qual['s'], qual['corner_weight'], lw=1.25, ls='--', label='corner weight')
    ax2.plot(qual['s'], qual['frame_overlap'], lw=1.0, ls=':', color='tab:orange', label='label overlap')
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', labelcolor='0.25')
    ax2.text(1.008, 0.5, 'weight / overlap', transform=ax2.transAxes, rotation=90, fontsize=6.8, color='0.25', va='center', ha='left')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', frameon=False, fontsize=6.8)

    # (f)
    ax = axs[5]
    cats = [r'$C_x$', r'$C_y$', r'$C_{\rm diag}$']
    dvals = [summary['bbh']['loops']['C_x']['D_loc_strict'], summary['bbh']['loops']['C_y']['D_loc_strict'], summary['bbh']['loops']['C_diag']['D_loc_strict']]
    epvals = [summary['bbh']['loops']['C_x']['ep_mean'], summary['bbh']['loops']['C_y']['ep_mean'], summary['bbh']['loops']['C_diag']['ep_mean']]
    xx = np.arange(len(cats))
    ax.plot(xx, dvals, 'o-', lw=1.2, ms=4.5, label=r'$D_{\rm loc}$')
    ax.set_xticks(xx, cats)
    ax.set_ylabel(r'$D_{\rm loc}$')
    ax.set_xlim(-0.4, len(cats) - 0.6)
    ymin = min(dvals) - 0.01
    ymax = max(dvals) + 0.008
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(dvals):
        if i == len(dvals) - 1:
            ax.text(i, min(v - 0.004, ymax - 0.002), f'{v:.3f}', ha='center', va='top', fontsize=7.5)
        else:
            ax.text(i, min(v + 0.0035, ymax - 0.002), f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax2 = ax.twinx()
    ax2.plot(xx, epvals, 's--', color='tab:orange', lw=1.1, ms=3.8, label=r'$e_p$')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('')
    right_label_above(ax, r'$e_p$', color='tab:orange', x=1.01, y=1.015)
    ax.legend(ax.get_lines() + ax2.get_lines(), [r'$D_{\rm loc}$', r'$e_p$'], loc='upper left', frameon=False)

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_bbh')


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------


def draw_bhz():
    grid = pd.read_csv(data_file('bhz_torus_grid.csv'))
    qual = pd.read_csv(data_file('bhz_counterrotating_loop_quality.csv'))
    width_df = pd.read_csv(data_file('bhz_width_scaling.csv'))
    phase_top = pd.read_csv(data_file('bhz_top_loop_phases.csv'), index_col=0)['phase'].to_numpy()
    phase_diag = pd.read_csv(data_file('bhz_corotating_loop_phases.csv'), index_col=0)['phase'].to_numpy()
    phase_anti = pd.read_csv(data_file('bhz_counterrotating_loop_phases.csv'), index_col=0)['phase'].to_numpy()

    _, _, joint = pivot(grid, 'theta_T', 'theta_B', 'joint')
    _, _, nloc = pivot(grid, 'theta_T', 'theta_B', 'nloc')
    phase_mat = np.vstack([phase_top, phase_diag, phase_anti])

    fig = plt.figure(figsize=(7.35, 4.95))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.95, bottom=0.11, wspace=0.68, hspace=0.56)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    spread_row_axes(axs[3:], shifts=[-0.010, 0.004, 0.018], width_shrink=0.90)

    # (a)
    ax = axs[0]
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.16, 0.24), 0.68, 0.52, fc='#d7dce3', ec='black', lw=0.8))
    ax.add_patch(Rectangle((0.16, 0.24), 0.68, 0.10, fc='#e4c697', ec='none'))
    ax.add_patch(Rectangle((0.16, 0.66), 0.68, 0.10, fc='#c8d8e7', ec='none'))
    ax.annotate('', xy=(0.78, 0.50), xytext=(0.26, 0.50), arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.text(0.50, 0.60, r'$k_x = 0$', ha='center', fontsize=10)
    ax.text(0.34, 0.80, r'$\theta_T$', ha='center')
    ax.text(0.70, 0.16, r'$\theta_B$', ha='center')
    ax.text(0.5, 0.04, r'local qubits: edge $\otimes$ Kramers', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (b)
    ax = axs[1]
    im = ax.imshow(joint.T, origin='lower', extent=[0, 2 * math.pi, 0, 2 * math.pi], cmap='viridis', aspect='equal', interpolation='bilinear')
    ax.plot([0, 2 * math.pi], [0, 2 * math.pi], color='limegreen', lw=1.2)
    ax.plot([0, 2 * math.pi], [2 * math.pi, 0], color='red', lw=1.2)
    ax.set_xlabel(r'$\theta_T$')
    ax.set_ylabel(r'$\theta_B$')
    set_pi_ticks(ax)
    cb = add_cbar(fig, ax, im, label=r'$\epsilon_{\rm joint}$')
    cb.ax.tick_params(labelsize=7)

    # (c)
    ax = axs[2]
    im = ax.imshow(nloc.T, origin='lower', extent=[0, 2 * math.pi, 0, 2 * math.pi], cmap='viridis', aspect='equal', interpolation='bilinear')
    ax.plot([0, 2 * math.pi], [0, 2 * math.pi], color='limegreen', lw=1.2)
    ax.plot([0, 2 * math.pi], [2 * math.pi, 0], color='red', lw=1.2)
    ax.set_xlabel(r'$\theta_T$')
    ax.set_ylabel(r'$\theta_B$')
    set_pi_ticks(ax)
    cb = add_cbar(fig, ax, im, label=r'$N_{\rm loc}$')
    cb.ax.tick_params(labelsize=7)

    # (d)
    ax = axs[3]
    im = ax.imshow(phase_mat, cmap='coolwarm', vmin=-0.25, vmax=0.25, aspect='auto', interpolation='nearest')
    xlabels = [r'$T\!\downarrow$', r'$T\!\uparrow$', r'$B\!\downarrow$', r'$B\!\uparrow$']
    ylabels = [r'$C_T$', r'$C_+$', r'$C_-$']
    ax.set_xticks(range(4), labels=xlabels)
    ax.set_yticks(range(3), labels=ylabels)
    for i in range(phase_mat.shape[0]):
        for j in range(phase_mat.shape[1]):
            val = phase_mat[i, j]
            color = 'white' if abs(val) > 0.13 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.6, color=color, path_effects=STROKE_BLACK if color == 'white' else None)
    add_cbar(fig, ax, im, pad=0.012, label=None)

    # (e)
    ax = axs[4]
    x = 2.0 * qual['s']
    ax.plot(x, qual['gap'], lw=1.25, label=r'$\Delta_4$')
    ax.plot(x, qual['deltaA'], lw=1.25, label=r'$\delta_A$')
    ax.plot(x, qual['deltaB'], lw=1.25, label=r'$\delta_B$')
    ax.set_xlabel(r'$t/\pi$')
    ax.set_ylabel('gap / split scale')
    ax.set_xlim(0, 2.0)
    ax2 = ax.twinx()
    ax2.plot(x, 100.0 * qual['joint'], lw=1.25, ls='--', color='tab:purple', label=r'$100\,\epsilon_{\rm joint}$')
    ax2.plot(x, qual['edge_weight_2row'], lw=1.0, ls=':', color='tab:brown', label='edge weight')
    ax2.set_ylabel('')
    ax2.text(1.008, 0.5, 'quality scale', transform=ax2.transAxes, rotation=90, fontsize=6.8, color='0.25', va='center', ha='left')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='lower right', frameon=False)

    # (f)
    ax = axs[5]
    width_df = width_df.copy()
    loop_map = {'top': (r'$C_T$', 'o-'), 'diag': (r'$C_+$', 's-'), 'anti': (r'$C_-$', '^-')}
    for loop, (lab, style) in loop_map.items():
        sub = width_df[width_df['loop'] == loop].sort_values('Ly')
        ax.plot(sub['Ly'], sub['D_loc'], style, lw=1.2, ms=4.5, label=lab)
    ax.set_xlabel(r'ribbon width $L_y$')
    ax.set_ylabel(r'$D_{\rm loc}$')
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.11), frameon=False)

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_bhz')


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------


def draw_bhz_annulus():
    grid = pd.read_csv(data_file('bhz_annulus_grid.csv'))
    curves = pd.read_csv(data_file('bhz_annulus_k_slices.csv'))
    x, y, gap = pivot(grid, 'kx', 'vartheta', 'gap')
    _, _, joint = pivot(grid, 'kx', 'vartheta', 'joint')
    _, _, nloc = pivot(grid, 'kx', 'vartheta', 'nloc_theta')

    fig = plt.figure(figsize=(7.35, 4.95))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.97, top=0.95, bottom=0.11, wspace=0.62, hspace=0.56)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # (a)
    ax = axs[0]
    ax.set_axis_off()
    ann = Wedge((0.50, 0.52), 0.33, 20, 340, width=0.13, fc='0.93', ec='black', lw=1.1)
    ax.add_patch(ann)
    ax.annotate('', xy=(0.36, 0.76), xytext=(0.48, 0.82), arrowprops=dict(arrowstyle='->', lw=1.0, connectionstyle='arc3,rad=0.55'))
    ax.text(0.27, 0.78, r'increasing $\vartheta$', ha='center')
    ax.annotate('', xy=(0.90, 0.50), xytext=(0.74, 0.50), arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.text(0.96, 0.50, r'$k_x$', ha='left', va='center', fontsize=10)
    ax.text(0.33, 0.90, r'$0.05\leq k_x\leq 0.35$', ha='center', fontsize=9.5)
    ax.text(0.50, 0.15, r'$(\theta_T,\theta_B)=(\vartheta,-\vartheta)$', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (b)
    ax = axs[1]
    im = annulus_heatmap(ax, x, y, gap, cmap='magma')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$\vartheta$')
    ax.set_xticks([0.05, 0.20, 0.35], [r'$0.05$', r'$0.2$', r'$0.35$'])
    ax.set_yticks(PI_TICKS, PI_TICKLABELS)
    cb = add_cbar(fig, ax, im)
    set_decimal_colorbar(cb, [0.35, 0.40, 0.45], digits=2)

    # (c)
    ax = axs[2]
    im = annulus_heatmap(ax, x, y, joint, cmap='viridis')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$\vartheta$')
    ax.set_xticks([0.05, 0.20, 0.35], [r'$0.05$', r'$0.2$', r'$0.35$'])
    ax.set_yticks(PI_TICKS, PI_TICKLABELS)
    cb = add_cbar(fig, ax, im)
    set_decimal_colorbar(cb, [0.01, 0.015, 0.02], digits=3)

    # (d)
    ax = axs[3]
    im = annulus_heatmap(ax, x, y, nloc, cmap='plasma')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$\vartheta$')
    ax.set_xticks([0.05, 0.20, 0.35], [r'$0.05$', r'$0.2$', r'$0.35$'])
    ax.set_yticks(PI_TICKS, PI_TICKLABELS)
    cb = add_cbar(fig, ax, im)
    set_decimal_colorbar(cb, [0.055, 0.060, 0.065, 0.070], digits=3)

    # (e)
    ax = axs[4]
    ax.plot(curves['kx'], curves['D_top'], lw=1.3, label=r'$C_T$')
    ax.plot(curves['kx'], curves['D_diag'], lw=1.3, label=r'$C_+$')
    ax.plot(curves['kx'], curves['D_anti'], lw=1.3, label=r'$C_-$')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$D_{\rm loc}$')
    ax2 = ax.twinx()
    ax2.plot(curves['kx'], curves['anti_fit_ZZ'], lw=1.2, ls='--', color='black', label=r'fit to $Z\otimes Z$')
    ax2.set_ylabel('')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', frameon=False)

    # (f)
    ax = axs[5]
    ax.plot(curves['kx'], curves['Cout_top_plusplus'], lw=1.3, label=r'$C_T$')
    ax.plot(curves['kx'], curves['Cout_diag_plusplus'], lw=1.3, label=r'$C_+$')
    ax.plot(curves['kx'], curves['Cout_anti_plusplus'], lw=1.3, label=r'$C_-$')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel('concurrence')
    ax.legend(loc='upper left', frameon=False)

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_bhz_annulus')


# -----------------------------------------------------------------------------
# Figure 6
# -----------------------------------------------------------------------------

def coeff_matrix(gen_df: pd.DataFrame, loop: str) -> np.ndarray:
    order = ['I', 'X', 'Y', 'Z']
    arr = np.zeros((4, 4))
    sub = gen_df[gen_df['loop'] == loop]
    for _, r in sub.iterrows():
        arr[order.index(r['pauli_left']), order.index(r['pauli_right'])] = float(r['coefficient'])
    return arr


def draw_mechanism():
    gen_df = pd.read_csv(data_file('fitted_generator_matrices.csv'))
    loop_family = pd.read_csv(data_file('continuous_loop_scan.csv'))

    fig = plt.figure(figsize=(7.35, 4.90))
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.955, top=0.95, bottom=0.12, wspace=0.70, hspace=0.56)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    for ax, loop in zip(axs[:3], ['SSH_left', 'BBH_diag', 'BHZ_anti']):
        arr = coeff_matrix(gen_df, loop)
        vmax = np.max(np.abs(arr))
        im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(4), labels=[r'$I$', r'$X$', r'$Y$', r'$Z$'])
        ax.set_yticks(range(4), labels=[r'$I$', r'$X$', r'$Y$', r'$Z$'])
        ax.set_xlabel('qubit B')
        ax.set_ylabel('qubit A')
        for i in range(4):
            for j in range(4):
                val = arr[i, j]
                if abs(val) > 0.02 * vmax:
                    color = 'white' if abs(val) > 0.55 * vmax else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color, path_effects=STROKE_BLACK if color == 'white' else None)
        add_cbar(fig, ax, im, pad=0.008)

    for ax, model in zip(axs[3:], ['SSH', 'BBH', 'BHZ']):
        sub = loop_family[(loop_family['model'] == model) & (loop_family['eta_numeric'].notna())].sort_values('eta_numeric')
        axis2 = loop_family[(loop_family['model'] == model) & (loop_family['eta_numeric'].isna())]['D_loc'].iloc[0]
        ax.plot(sub['eta_numeric'], sub['D_loc'], color='tab:blue', lw=1.35)
        ax.axhline(axis2, color='0.45', lw=0.9, ls=':')
        ax.axvline(-1, color='0.75', lw=0.8, ls='--')
        ax.axvline(0, color='0.8', lw=0.8, ls='--')
        ax.axvline(1, color='0.75', lw=0.8, ls='--')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$D_{\rm loc}$')
        ax.set_xlim(-2.0, 2.0)
        ax2 = ax.twinx()
        ax2.plot(sub['eta_numeric'], sub['ep_mean'], color='tab:orange', lw=1.0)
        ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=7)
        ax2.set_ylabel('')
        right_label_above(ax, r'$e_p$', color='tab:orange')
        ax.text(0.03, 0.90, '2nd-axis control', transform=ax.transAxes, fontsize=6.7, color='0.35')

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_mechanism')


# -----------------------------------------------------------------------------
# Figure 8
# -----------------------------------------------------------------------------

def draw_controls():
    summary = load_results_summary()
    ssh_b = pd.read_csv(data_file('ssh_small_field_scan.csv'))
    disorder = pd.read_csv(data_file('bbh_disorder_scan.csv'))
    trivial = pd.read_csv(data_file('control_diagnostics.csv'))

    fit = summary['ssh']['small_B_fit']

    fig = plt.figure(figsize=(7.35, 4.95))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.97, top=0.95, bottom=0.12, wspace=0.32, hspace=0.40)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # (a)
    ax = axs[0]
    Bfit = np.linspace(0.0, 0.40, 200)
    ax.plot(ssh_b['B'], ssh_b['D_loc'], 'o-', lw=1.3, label=r'$D_{\rm loc}$')
    ax.plot(Bfit, fit['D_loc_coeff'] * Bfit ** 2 + fit['D_loc_intercept'], '--', lw=1.1, label=fit_label_math(fit['D_loc_coeff'], 2, fit['D_loc_intercept']))
    ax.set_xlabel(r'$B$')
    ax.set_ylabel(r'$D_{\rm loc}(C_L)$')
    ax.legend(loc='upper left', frameon=False)

    # (b)
    ax = axs[1]
    ax.plot(ssh_b['B'], ssh_b['ep_mean'], 'o-', lw=1.3, label=r'$e_p$')
    ax.plot(Bfit, fit['ep_coeff'] * Bfit ** 4 + fit['ep_intercept'], '--', lw=1.1, label=fit_label_math(fit['ep_coeff'], 4, fit['ep_intercept']))
    ax.set_xlabel(r'$B$')
    ax.set_ylabel(r'$e_p(C_L)$')
    ax.legend(loc='upper left', frameon=False)

    # (c)
    ax = axs[2]
    ax.errorbar(disorder['W'], disorder['D_loc_mean'], yerr=disorder['D_loc_std'], fmt='o-', lw=1.3, capsize=0, label=r'$D_{\rm loc}$')
    ax.set_xlabel(r'$W$')
    ax.set_ylabel(r'$D_{\rm loc}(C_{\rm diag})$')
    ax2 = ax.twinx()
    ax2.errorbar(disorder['W'], disorder['gap_mean'], yerr=disorder['gap_std'], fmt='s-', lw=1.3, capsize=0, color='tab:orange', label=r'$\Delta_{4,\min}$')
    ax2.set_ylabel(r'$\Delta_{4,\min}$', labelpad=1)
    ax2.yaxis.set_label_coords(1.02, 0.5)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [r'$D_{\rm loc}$', r'$\Delta_{4,\min}$']
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.86, 1.02), frameon=False)

    # (d)
    ax = axs[3]
    cats = ['SSH top.', 'SSH triv.', 'BBH HOTI', 'BBH triv.']
    local_weight = [
        summary['ssh']['edge_weight_min'],
        float(trivial[trivial['model'] == 'SSH_trivial']['local_weight'].iloc[0]),
        summary['bbh']['diag_quality']['corner_weight_min'],
        float(trivial[trivial['model'] == 'BBH_trivial']['local_weight'].iloc[0]),
    ]
    max_joint = [
        summary['ssh']['joint_max'],
        float(trivial[trivial['model'] == 'SSH_trivial']['joint_max'].iloc[0]),
        summary['bbh']['diag_quality']['joint_max'],
        float(trivial[trivial['model'] == 'BBH_trivial']['joint_max'].iloc[0]),
    ]
    xx = np.arange(len(cats))
    w = 0.34
    ax.bar(xx - w / 2, local_weight, width=w, label='local weight')
    ax.bar(xx + w / 2, max_joint, width=w, label=r'max $\epsilon_{\rm joint}$')
    ax.set_xticks(xx, cats, rotation=15)
    ax.set_ylabel('quality metric')
    ax.legend(loc='upper right', frameon=False)

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)', '(d)'], dx=0.018, dy=0.008)
    savefig(fig, 'fig_controls')




# -----------------------------------------------------------------------------
# Figure 9
# -----------------------------------------------------------------------------

def draw_robustness():
    robust = pd.read_csv(data_file('observable_robustness_scan.csv'))

    fig, axs = plt.subplots(1, 3, figsize=(7.35, 2.35))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.24, wspace=0.44)

    panel_specs = [
        (
            'SSH',
            ['reference', 'edge_variant', 'spin_variant', 'combined_variant'],
            ['reference', 'edge\nvariant', 'spin\nvariant', 'combined\nvariant'],
            [('left', r'$C_L$'), ('diag', r'$C_{\rm diag}$')],
            (0.0, 0.205),
        ),
        (
            'BBH',
            ['reference', 'cubic_variant'],
            ['reference', 'cubic\nvariant'],
            [('x', r'$C_x$'), ('diag', r'$C_{\rm diag}$')],
            (0.0, 0.155),
        ),
        (
            'BHZ',
            ['reference', 'side_variant', 'spin_variant', 'combined_variant'],
            ['reference', 'side\nvariant', 'spin\nvariant', 'combined\nvariant'],
            [('diag', r'$C_+$'), ('anti', r'$C_-$')],
            (0.0, 0.385),
        ),
    ]

    for ax, (model, variants, labels, loops, ylim) in zip(axs, panel_specs):
        sub = robust[robust['model'] == model]
        xx = np.arange(len(variants))
        for loop_key, loop_label in loops:
            vals = []
            for variant in variants:
                row = sub[(sub['variant'] == variant) & (sub['loop'] == loop_key)]
                vals.append(float(row['D_loc'].iloc[0]))
            ax.plot(xx, vals, 'o-', lw=1.3, label=loop_label)
        ax.set_xticks(xx, labels)
        ax.set_ylim(*ylim)
        ax.set_ylabel(r'$D_{\rm loc}$')
        ax.legend(frameon=False, loc='upper right')

    add_panel_labels(fig, axs, ['(a)', '(b)', '(c)'], dx=0.016, dy=0.008)
    savefig(fig, 'fig_robustness')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ensure_additional_datasets()
    draw_pipeline()
    draw_ssh()
    draw_bbh()
    draw_bhz()
    draw_bhz_annulus()
    draw_mechanism()
    draw_controls()
    draw_robustness()
    print('Built the figure files.')


if __name__ == '__main__':
    main()
