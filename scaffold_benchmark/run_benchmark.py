"""
Main benchmark runner.
Runs Simple GA, NSGA-II, and BO × 20 seeds each, computes metrics,
generates figures and data files.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

from .simple_ga import run_simple_ga
from .nsga2_runner import run_nsga2, run_nsga2_reference
from .bo_runner import run_bo
from .problem import compressive_stiffness, porosity_score, is_feasible, BOUNDS
from . import metrics
from . import visualise

SEEDS = list(range(20))
RESULTS_DIR = Path('results')
FIG_DIR = RESULTS_DIR / 'figures'
DATA_DIR = RESULTS_DIR / 'data'


def _run_nsga2_with_ref(args):
    """Wrapper so we can pass ref_point via process_map.

    Parameters:
        args (tuple): (seed, ref_point).
    Returns:
        dict: NSGA-II result.
    """
    seed, ref_point = args
    return run_nsga2(seed, n_gen=150, ref_point=ref_point)


def main():
    """Entry point for the full benchmark."""
    from tqdm.contrib.concurrent import process_map

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    use_parallel = os.cpu_count() is not None and os.cpu_count() > 2
    max_workers = max(1, (os.cpu_count() or 1) - 1) if use_parallel else 1

    # ──────────────────────────────────────────────────────────
    # Step 0: Reference Pareto front (NSGA-II, 500 gens)
    # ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Generating NSGA-II reference Pareto front (500 gens)...")
    print("=" * 60)
    t0 = time.time()
    ref_front = run_nsga2_reference(seed=42, n_gen=500)
    ref_point = ref_front.max(axis=0) * 1.1 if ref_front.shape[0] > 0 else np.array([0.0, 0.0])
    print(f"  Reference front: {ref_front.shape[0]} points  ({time.time()-t0:.1f}s)")
    print(f"  HV ref point: {ref_point}")

    # ──────────────────────────────────────────────────────────
    # Step 1: Run Simple GA × 20
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running Simple GA (20 seeds)...")
    print("=" * 60)
    if use_parallel:
        ga_results = process_map(run_simple_ga, SEEDS, max_workers=max_workers,
                                  desc='Simple GA', chunksize=1)
    else:
        from tqdm import tqdm
        ga_results = [run_simple_ga(s) for s in tqdm(SEEDS, desc='Simple GA')]

    # ──────────────────────────────────────────────────────────
    # Step 2: Run NSGA-II × 20
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running NSGA-II (20 seeds)...")
    print("=" * 60)
    nsga2_args = [(s, ref_point) for s in SEEDS]
    if use_parallel:
        nsga2_results = process_map(_run_nsga2_with_ref, nsga2_args,
                                     max_workers=max_workers, desc='NSGA-II', chunksize=1)
    else:
        from tqdm import tqdm
        nsga2_results = [_run_nsga2_with_ref(a) for a in tqdm(nsga2_args, desc='NSGA-II')]

    # ──────────────────────────────────────────────────────────
    # Step 3: Run BO × 20
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running Bayesian Optimisation (20 seeds)...")
    print("=" * 60)
    # BO must run sequentially due to module-level _current_seed
    from tqdm import tqdm
    bo_results = [run_bo(s) for s in tqdm(SEEDS, desc='BO')]

    # ──────────────────────────────────────────────────────────
    # Step 4: Compute metrics
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    all_results = {
        'SimpleGA': ga_results,
        'NSGA2': nsga2_results,
        'BO': bo_results,
    }

    # M2: Final best stats
    ga_stats = metrics.final_best_stats(ga_results)
    nsga_stats = metrics.final_best_stats(nsga2_results)
    bo_stats = metrics.final_best_stats(bo_results)

    # M3: 90% target (dynamic)
    target_90 = metrics.compute_90pct_target(all_results)
    print(f"  90% target stiffness: {target_90:.2f} MPa")

    # Build stiffness-based eval_history for GA (convert fitness to stiffness approx)
    # GA eval_history has (eval#, best_fitness). Best_fitness = stiffness - penalty.
    # For feasible solutions, best_fitness ≈ stiffness. We use it directly for M3.
    ga_evals_90 = metrics.evals_to_90pct(ga_results, target_90, key='eval_history')

    # BO eval_history has (eval#, best_so_far_obj) where obj = -stiffness + penalty
    # Convert: stiffness ≈ -obj for feasible. We need to compare stiffness >= target_90.
    # Build converted histories for BO
    for r in bo_results:
        r['stiffness_eval_history'] = [(ev, -val) for ev, val in r['eval_history']]
    bo_evals_90 = metrics.evals_to_90pct(bo_results, target_90, key='stiffness_eval_history')

    # NSGA-II: use best_stiffness directly (per-run, not per-eval)
    nsga_evals_90 = []
    for r in nsga2_results:
        if r['best_stiffness'] >= target_90:
            nsga_evals_90.append(r.get('n_evals', 15000))
        else:
            nsga_evals_90.append(np.inf)

    # M4: Constraint satisfaction
    ga_feas = metrics.constraint_satisfaction_rate(ga_results)
    bo_feas = metrics.constraint_satisfaction_rate(bo_results)
    # NSGA-II handles constraints natively
    nsga_feas = 100.0  # pymoo enforces feasibility

    # M5: Hypervolume
    hv_data = metrics.hypervolume_per_gen(nsga2_results, ref_point)

    # M6: Pareto front size
    pf_size = metrics.pareto_front_size(nsga2_results)

    # M7: GD/IGD
    gd_igd = metrics.gd_igd_metrics(nsga2_results, ref_front)

    # Compute efficiency
    evals_dict = {'Simple GA': ga_evals_90, 'NSGA-II': nsga_evals_90, 'BO': bo_evals_90}
    efficiency = metrics.compute_efficiency(evals_dict)

    # ──────────────────────────────────────────────────────────
    # Step 5: Generate figures
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    visualise.fig1_convergence_curves(ga_results, nsga2_results, bo_results, FIG_DIR)
    print("  [OK] fig1_convergence_curves.png")

    visualise.fig2_pareto_front_nsga2(nsga2_results, FIG_DIR)
    print("  [OK] fig2_pareto_front_nsga2.png")

    stat_info = visualise.fig3_boxplots_final_quality(ga_results, nsga2_results, bo_results, FIG_DIR)
    print("  [OK] fig3_boxplots_final_quality.png")

    visualise.fig4_design_space_scatter(ga_results, nsga2_results, bo_results, FIG_DIR)
    print("  [OK] fig4_design_space_scatter.png")

    # Build radar metrics (normalise 0-1)
    def _norm(vals):
        mn, mx = min(vals.values()), max(vals.values())
        if mx == mn:
            return {k: 1.0 for k in vals}
        return {k: (v - mn) / (mx - mn) for k, v in vals.items()}

    # Convergence speed: 1/mean_evals_to_90pct (higher is better)
    conv_speed_raw = {}
    for algo, ev_list in evals_dict.items():
        finite = [v for v in ev_list if np.isfinite(v)]
        conv_speed_raw[algo] = 1.0 / np.mean(finite) if finite else 0.0
    conv_speed = _norm(conv_speed_raw)

    # Final quality: mean best stiffness
    fq = _norm({'Simple GA': ga_stats['mean'], 'NSGA-II': nsga_stats['mean'], 'BO': bo_stats['mean']})

    # Diversity: for NSGA-II use PF size, for others use 1 (single solution)
    div_raw = {'Simple GA': 1.0, 'NSGA-II': pf_size['mean'], 'BO': 1.0}
    diversity = _norm(div_raw)

    # Constraint satisfaction
    cs = _norm({'Simple GA': ga_feas, 'NSGA-II': nsga_feas, 'BO': bo_feas})

    radar_data = {}
    for algo in ['Simple GA', 'NSGA-II', 'BO']:
        radar_data[algo] = {
            'convergence_speed': conv_speed[algo],
            'final_quality': fq[algo],
            'diversity': diversity[algo],
            'constraint_satisfaction': cs[algo],
            'compute_efficiency': efficiency[algo],
        }

    visualise.fig5_radar_chart(radar_data, FIG_DIR)
    print("  [OK] fig5_radar_chart.png")

    # --------------------------------------------------------------
    # Step 6: Save data files
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving data files...")
    print("=" * 60)

    # results_summary.csv
    rows = []
    for r in ga_results:
        rows.append({
            'algorithm': 'SimpleGA', 'run_seed': r['seed'],
            'best_stiffness': r['best_stiffness'], 'best_porosity': r['best_porosity'],
            'n_evals_to_90pct': ga_evals_90[ga_results.index(r)],
            'is_feasible': r['is_feasible'], 'final_hypervolume': np.nan,
        })
    for i, r in enumerate(nsga2_results):
        hv_final = r['hv_history'][-1] if r['hv_history'] else np.nan
        rows.append({
            'algorithm': 'NSGA2', 'run_seed': r['seed'],
            'best_stiffness': r['best_stiffness'], 'best_porosity': np.nan,
            'n_evals_to_90pct': nsga_evals_90[i],
            'is_feasible': True, 'final_hypervolume': hv_final,
        })
    for i, r in enumerate(bo_results):
        rows.append({
            'algorithm': 'BO', 'run_seed': r['seed'],
            'best_stiffness': r['best_stiffness'], 'best_porosity': r['best_porosity'],
            'n_evals_to_90pct': bo_evals_90[i],
            'is_feasible': r['is_feasible'], 'final_hypervolume': np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'results_summary.csv', index=False)
    print("  [OK] results_summary.csv")

    # pareto_front_best_run.csv
    best_nsga_idx = 0
    best_hv = 0
    for i, r in enumerate(nsga2_results):
        if r['hv_history'] and r['hv_history'][-1] > best_hv:
            best_hv = r['hv_history'][-1]
            best_nsga_idx = i
    best_run = nsga2_results[best_nsga_idx]
    pf_df = pd.DataFrame(best_run['pareto_X'], columns=['wall_thickness', 'unit_cell_size',
                                                          'porosity_fraction', 'strut_angle'])
    pf_df['stiffness_MPa'] = -best_run['pareto_F'][:, 0]
    pf_df['porosity'] = -best_run['pareto_F'][:, 1]
    pf_df.to_csv(DATA_DIR / 'pareto_front_best_run.csv', index=False)
    print("  [OK] pareto_front_best_run.csv")

    # statistical_tests.txt
    from scipy import stats as sp_stats
    lines = ["Statistical Tests Summary", "=" * 40, ""]
    H, p_kw = sp_stats.kruskal(
        [r['best_stiffness'] for r in ga_results],
        [r['best_stiffness'] for r in nsga2_results],
        [r['best_stiffness'] for r in bo_results],
    )
    lines.append(f"Kruskal-Wallis H={H:.4f}, p={p_kw:.6f}")
    lines.append("")
    pairs = [
        ('SimpleGA', 'NSGA2', ga_results, nsga2_results),
        ('SimpleGA', 'BO', ga_results, bo_results),
        ('NSGA2', 'BO', nsga2_results, bo_results),
    ]
    for n1, n2, r1, r2 in pairs:
        v1 = [r['best_stiffness'] for r in r1]
        v2 = [r['best_stiffness'] for r in r2]
        u, p = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_corr = min(p * 3, 1.0)
        sig = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
        lines.append(f"  {n1} vs {n2}: U={u:.1f}, p_corrected={p_corr:.6f} ({sig})")

    lines.append("")
    lines.append("GD/IGD Metrics (NSGA-II):")
    lines.append(f"  GD:  {gd_igd['gd_mean']:.6f} +/- {gd_igd['gd_std']:.6f}")
    lines.append(f"  IGD: {gd_igd['igd_mean']:.6f} +/- {gd_igd['igd_std']:.6f}")
    lines.append("")
    lines.append(f"Pareto front size: {pf_size['mean']:.1f} +/- {pf_size['std']:.1f}")
    lines.append(f"Final HV (mean +/- std): {hv_data['mean'][-1]:.4f} +/- {hv_data['std'][-1]:.4f}")

    (DATA_DIR / 'statistical_tests.txt').write_text('\n'.join(lines), encoding='utf-8')
    print("  [OK] statistical_tests.txt")

    # --------------------------------------------------------------
    # Step 7: Print summary table
    # --------------------------------------------------------------
    def _mean_finite(lst):
        f = [v for v in lst if np.isfinite(v)]
        return f"{np.mean(f):.0f}" if f else "N/A"

    hv_final_mean = hv_data['mean'][-1] if len(hv_data['mean']) > 0 else 0
    hv_final_std = hv_data['std'][-1] if len(hv_data['std']) > 0 else 0

    print("\n")
    print("+---------------------+------------+------------+------------+")
    print("| Metric              | Simple GA  |  NSGA-II   |    BO      |")
    print("+---------------------+------------+------------+------------+")
    print(f"| Best stiffness(MPa) | {ga_stats['mean']:>6.1f}+/-{ga_stats['std']:<3.1f}| {nsga_stats['mean']:>6.1f}+/-{nsga_stats['std']:<3.1f}| {bo_stats['mean']:>6.1f}+/-{bo_stats['std']:<3.1f}|")
    print(f"| Evals to 90% target | {_mean_finite(ga_evals_90):>10s} | {_mean_finite(nsga_evals_90):>10s} | {_mean_finite(bo_evals_90):>10s} |")
    print(f"| Feasibility rate    | {ga_feas:>9.0f}% | {nsga_feas:>9.0f}% | {bo_feas:>9.0f}% |")
    print(f"| Final hypervolume   |    N/A     |{hv_final_mean:>6.3f}+/-{hv_final_std:<3.2f}|    N/A     |")
    print("+---------------------+------------+------------+------------+")
    print(f"\nAll results saved to {RESULTS_DIR.resolve()}")


if __name__ == '__main__':
    main()
