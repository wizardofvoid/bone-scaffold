"""
Publication-ready figure generation for the scaffold benchmark.
All figures saved as 300 dpi PNGs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from scipy import stats


def _setup_style():
    """Apply consistent publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })


def fig1_convergence_curves(ga_results, nsga2_results, bo_results, out_dir):
    """3-panel convergence curves with log x-axis and ±1 std shading.

    Parameters:
        ga_results (list[dict]): SimpleGA run results.
        nsga2_results (list[dict]): NSGA-II run results.
        bo_results (list[dict]): BO run results.
        out_dir (Path): Output directory for figures.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # --- Simple GA ---
    ax = axes[0]
    max_evals_ga = max(len(r['eval_history']) for r in ga_results)
    mat_ga = np.full((len(ga_results), max_evals_ga), np.nan)
    for i, r in enumerate(ga_results):
        vals = [v for _, v in r['eval_history']]
        mat_ga[i, :len(vals)] = vals
    for i in range(mat_ga.shape[0]):
        for j in range(1, mat_ga.shape[1]):
            if np.isnan(mat_ga[i, j]):
                mat_ga[i, j] = mat_ga[i, j - 1]
    evals_ga = np.arange(1, max_evals_ga + 1)
    mean_ga = np.nanmean(mat_ga, axis=0)
    std_ga = np.nanstd(mat_ga, axis=0)
    ax.plot(evals_ga, mean_ga, color='#2196F3', linewidth=1.2)
    ax.fill_between(evals_ga, mean_ga - std_ga, mean_ga + std_ga, alpha=0.25, color='#2196F3')
    ax.axvline(500, linestyle='--', color='grey', alpha=0.6, linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Simple GA')

    # --- NSGA-II (best stiffness = -min(f1)) ---
    ax = axes[1]
    max_gen = max(len(r['hv_history']) for r in nsga2_results)
    # For NSGA-II convergence, track best stiffness per gen is not directly available
    # Use HV history instead as the convergence metric
    mat_nsga = np.full((len(nsga2_results), max_gen), np.nan)
    for i, r in enumerate(nsga2_results):
        h = r['hv_history']
        mat_nsga[i, :len(h)] = h
    for i in range(mat_nsga.shape[0]):
        for j in range(1, mat_nsga.shape[1]):
            if np.isnan(mat_nsga[i, j]):
                mat_nsga[i, j] = mat_nsga[i, j - 1]
    evals_nsga = np.arange(1, max_gen + 1) * 100  # 100 evals per gen
    mean_nsga = np.nanmean(mat_nsga, axis=0)
    std_nsga = np.nanstd(mat_nsga, axis=0)
    ax.plot(evals_nsga, mean_nsga, color='#4CAF50', linewidth=1.2)
    ax.fill_between(evals_nsga, mean_nsga - std_nsga, mean_nsga + std_nsga, alpha=0.25, color='#4CAF50')
    ax.axvline(500, linestyle='--', color='grey', alpha=0.6, linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Function Evaluations')
    ax.set_title('NSGA-II (Hypervolume)')

    # --- BO ---
    ax = axes[2]
    max_evals_bo = max(len(r['best_per_call']) for r in bo_results)
    mat_bo = np.full((len(bo_results), max_evals_bo), np.nan)
    for i, r in enumerate(bo_results):
        # BO minimises -stiff+penalty, so negate for "best fitness"
        vals = [-v for v in r['best_per_call']]
        mat_bo[i, :len(vals)] = vals
    for i in range(mat_bo.shape[0]):
        for j in range(1, mat_bo.shape[1]):
            if np.isnan(mat_bo[i, j]):
                mat_bo[i, j] = mat_bo[i, j - 1]
    evals_bo = np.arange(1, max_evals_bo + 1)
    mean_bo = np.nanmean(mat_bo, axis=0)
    std_bo = np.nanstd(mat_bo, axis=0)
    ax.plot(evals_bo, mean_bo, color='#FF9800', linewidth=1.2)
    ax.fill_between(evals_bo, mean_bo - std_bo, mean_bo + std_bo, alpha=0.25, color='#FF9800')
    ax.axvline(500, linestyle='--', color='grey', alpha=0.6, linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Function Evaluations')
    ax.set_title('Bayesian Optimisation')

    plt.tight_layout()
    fig.savefig(out_dir / 'fig1_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig2_pareto_front_nsga2(nsga2_results, out_dir):
    """Pareto front scatter: all runs grey, best run coloured, annotated.

    Parameters:
        nsga2_results (list[dict]): NSGA-II run results.
        out_dir (Path): Output directory.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Find best run by HV
    best_idx = 0
    best_hv = 0
    for i, r in enumerate(nsga2_results):
        if r['hv_history'] and r['hv_history'][-1] > best_hv:
            best_hv = r['hv_history'][-1]
            best_idx = i

    # Plot all runs in grey
    for i, r in enumerate(nsga2_results):
        F = r['pareto_F']
        if F.shape[0] > 0:
            ax.scatter(-F[:, 0], -F[:, 1], c='lightgrey', s=10, alpha=0.4, zorder=1)

    # Plot best run in colour
    F_best = nsga2_results[best_idx]['pareto_F']
    stiffness = -F_best[:, 0]
    porosity = -F_best[:, 1]
    sc = ax.scatter(stiffness, porosity, c=stiffness, cmap='viridis', s=30, edgecolors='k',
                    linewidths=0.3, zorder=2)
    plt.colorbar(sc, ax=ax, label='Stiffness (MPa)')

    # Annotate key points
    if len(stiffness) > 0:
        # Max stiffness
        idx_ms = np.argmax(stiffness)
        ax.annotate('Max Stiffness', (stiffness[idx_ms], porosity[idx_ms]),
                     textcoords='offset points', xytext=(10, -15), fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='red'), color='red')
        # Max porosity
        idx_mp = np.argmax(porosity)
        ax.annotate('Max Porosity', (stiffness[idx_mp], porosity[idx_mp]),
                     textcoords='offset points', xytext=(10, 10), fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        # Knee point (closest to utopia)
        utopia = np.array([np.max(stiffness), np.max(porosity)])
        dists = np.sqrt((stiffness - utopia[0]) ** 2 + (porosity - utopia[1]) ** 2)
        idx_knee = np.argmin(dists)
        ax.annotate('Knee Point', (stiffness[idx_knee], porosity[idx_knee]),
                     textcoords='offset points', xytext=(-15, -20), fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='green'), color='green')

    ax.set_xlabel('Compressive Stiffness (MPa)')
    ax.set_ylabel('Porosity (dimensionless)')
    ax.set_title('NSGA-II Pareto Front (Best Run Highlighted)')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig2_pareto_front_nsga2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig3_boxplots_final_quality(ga_results, nsga2_results, bo_results, out_dir):
    """Boxplots + strip plot with statistical significance annotations.

    Parameters:
        ga_results, nsga2_results, bo_results: Run result lists.
        out_dir (Path): Output directory.
    """
    _setup_style()

    ga_vals = [r['best_stiffness'] for r in ga_results]
    nsga_vals = [r['best_stiffness'] for r in nsga2_results]
    bo_vals = [r['best_stiffness'] for r in bo_results]

    import pandas as pd
    df = pd.DataFrame({
        'Stiffness': ga_vals + nsga_vals + bo_vals,
        'Algorithm': ['Simple GA'] * len(ga_vals) + ['NSGA-II'] * len(nsga_vals) + ['BO'] * len(bo_vals)
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {'Simple GA': '#2196F3', 'NSGA-II': '#4CAF50', 'BO': '#FF9800'}
    sns.boxplot(data=df, x='Algorithm', y='Stiffness', hue='Algorithm', palette=palette, ax=ax, width=0.5, legend=False)
    sns.stripplot(data=df, x='Algorithm', y='Stiffness', color='k', alpha=0.5, size=4, jitter=True, ax=ax)

    # Statistical tests
    H_stat, p_kw = stats.kruskal(ga_vals, nsga_vals, bo_vals)
    pairs = [('Simple GA', 'NSGA-II', ga_vals, nsga_vals),
             ('Simple GA', 'BO', ga_vals, bo_vals),
             ('NSGA-II', 'BO', nsga_vals, bo_vals)]
    sig_text = f'Kruskal-Wallis: H={H_stat:.2f}, p={p_kw:.4f}\n'
    for n1, n2, v1, v2 in pairs:
        u_stat, p_mw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_corr = min(p_mw * 3, 1.0)  # Bonferroni
        sig_text += f'{n1} vs {n2}: p={p_corr:.4f}'
        if p_corr < 0.05:
            sig_text += ' *'
        sig_text += '\n'

    ax.text(0.02, 0.98, sig_text.strip(), transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Best Stiffness (MPa)')
    ax.set_title('Final Best Stiffness Across 20 Runs')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig3_boxplots_final_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {'H_stat': H_stat, 'p_kw': p_kw, 'pairs': pairs, 'sig_text': sig_text}


def fig4_design_space_scatter(ga_results, nsga2_results, bo_results, out_dir):
    """2x2 grid of scatter plots for design variable pairs.

    Parameters:
        ga_results, nsga2_results, bo_results: Run result lists.
        out_dir (Path): Output directory.
    """
    _setup_style()
    var_names = ['wall_thickness', 'unit_cell_size', 'porosity_fraction', 'strut_angle']
    pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]

    def _collect_single(results, key):
        """Collect single-solution results (GA/BO) into array."""
        return np.array([r[key] for r in results if key in r])

    def _collect_multi(results, key):
        """Collect multi-solution results (NSGA-II Pareto) into array."""
        arrays = [r[key] for r in results if key in r and r[key].shape[0] > 0]
        return np.vstack(arrays) if arrays else np.empty((0, 4))

    ga_pts = _collect_single(ga_results, 'final_best')
    nsga_pts = _collect_multi(nsga2_results, 'pareto_X')
    bo_pts = _collect_single(bo_results, 'final_best_x')

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    colors = {'Simple GA': '#2196F3', 'NSGA-II': '#4CAF50', 'BO': '#FF9800'}
    markers = {'Simple GA': 'o', 'NSGA-II': 's', 'BO': '^'}

    for ax, (i, j) in zip(axes.flat, pairs):
        if ga_pts.shape[0] > 0:
            ax.scatter(ga_pts[:, i], ga_pts[:, j], c=colors['Simple GA'], marker='o',
                       s=25, alpha=0.6, label='Simple GA')
        if nsga_pts.shape[0] > 0:
            ax.scatter(nsga_pts[:, i], nsga_pts[:, j], c=colors['NSGA-II'], marker='s',
                       s=15, alpha=0.3, label='NSGA-II')
        if bo_pts.shape[0] > 0:
            ax.scatter(bo_pts[:, i], bo_pts[:, j], c=colors['BO'], marker='^',
                       s=25, alpha=0.6, label='BO')
        ax.set_xlabel(var_names[i])
        ax.set_ylabel(var_names[j])
        ax.legend(fontsize=7, loc='best')

    fig.suptitle('Design Space Exploration', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig4_design_space_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig5_radar_chart(metrics_dict, out_dir):
    """Radar chart comparing 5 normalised metrics across 3 algorithms.

    Parameters:
        metrics_dict (dict): {algo: {metric_name: value_0_to_1}}.
            Metrics: convergence_speed, final_quality, diversity,
                     constraint_satisfaction, compute_efficiency.
        out_dir (Path): Output directory.
    """
    _setup_style()
    categories = ['Convergence\nSpeed', 'Final\nQuality', 'Diversity',
                   'Constraint\nSatisfaction', 'Compute\nEfficiency']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = {'Simple GA': '#2196F3', 'NSGA-II': '#4CAF50', 'BO': '#FF9800'}

    for algo, vals in metrics_dict.items():
        values = [vals.get(k, 0) for k in
                  ['convergence_speed', 'final_quality', 'diversity',
                   'constraint_satisfaction', 'compute_efficiency']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=algo, color=colors.get(algo, 'grey'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(algo, 'grey'))

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1.05)
    ax.set_title('Algorithm Comparison', y=1.08, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    fig.savefig(out_dir / 'fig5_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
