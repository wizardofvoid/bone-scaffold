"""
Metrics computation for the benchmark (M1–M7 + compute_efficiency).
"""

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD


# ── M1: Convergence curves ─────────────────────────────────────────
def convergence_curves(results_list, key='eval_history'):
    """Mean ± std best-so-far at each function evaluation across runs.

    Parameters:
        results_list (list[dict]): Run results, each with eval_history [(eval#, best)].
        key (str): Key for evaluation history in each result dict.
    Returns:
        dict: 'evals' (1-D array), 'mean' (1-D), 'std' (1-D).
    """
    max_len = max(len(r[key]) for r in results_list)
    matrix = np.full((len(results_list), max_len), np.nan)
    for i, r in enumerate(results_list):
        vals = [v for _, v in r[key]]
        matrix[i, :len(vals)] = vals
    # Forward-fill NaNs
    for i in range(matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                matrix[i, j] = matrix[i, j - 1]
    evals = np.arange(1, max_len + 1)
    return {'evals': evals, 'mean': np.nanmean(matrix, axis=0), 'std': np.nanstd(matrix, axis=0)}


# ── M2: Final best stiffness statistics ────────────────────────────
def final_best_stats(results_list):
    """Mean, std, min, max of final best stiffness across runs.

    Parameters:
        results_list (list[dict]): Each must have 'best_stiffness'.
    Returns:
        dict: mean, std, min, max.
    """
    vals = np.array([r['best_stiffness'] for r in results_list])
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'min': float(np.min(vals)), 'max': float(np.max(vals))}


# ── M3: Evaluations to reach 90% of target ────────────────────────
def compute_90pct_target(all_results):
    """Compute the 90% target dynamically as 0.90 * max stiffness across all runs.

    Parameters:
        all_results (dict): {'SimpleGA': [...], 'NSGA2': [...], 'BO': [...]}
    Returns:
        float: 90% target stiffness value.
    """
    best_vals = []
    for algo, runs in all_results.items():
        for r in runs:
            best_vals.append(r['best_stiffness'])
    return 0.90 * max(best_vals)


def evals_to_90pct(results_list, target, key='eval_history'):
    """Number of function evaluations to reach target for each run.

    Parameters:
        results_list (list[dict]): Run results with eval_history.
        target (float): Stiffness target (90% of best).
        key (str): Key into result dicts.
    Returns:
        list[float]: Evals to reach target per run (np.inf if never reached).
    """
    out = []
    for r in results_list:
        found = np.inf
        for ev, val in r[key]:
            # For GA: val is best fitness (stiffness - penalty), approx stiffness if feasible
            # For BO: val is -stiffness + penalty (minimised), so stiffness ~ -val if feasible
            # We'll pass already-converted histories from the runner
            if val >= target:
                found = ev
                break
        out.append(found)
    return out


# ── M4: Constraint satisfaction rate ───────────────────────────────
def constraint_satisfaction_rate(results_list):
    """Percentage of final solutions that are feasible.

    Parameters:
        results_list (list[dict]): Each with 'is_feasible'.
    Returns:
        float: Percentage 0–100.
    """
    return 100.0 * sum(1 for r in results_list if r['is_feasible']) / len(results_list)


# ── M5: Hypervolume per generation (NSGA-II) ──────────────────────
def hypervolume_per_gen(nsga2_results, ref_point):
    """Mean ± std hypervolume per generation across NSGA-II runs.

    Parameters:
        nsga2_results (list[dict]): Each with 'hv_history'.
        ref_point (np.ndarray): Reference point for HV.
    Returns:
        dict: 'gens', 'mean', 'std'.
    """
    max_len = max(len(r['hv_history']) for r in nsga2_results)
    matrix = np.zeros((len(nsga2_results), max_len))
    for i, r in enumerate(nsga2_results):
        h = r['hv_history']
        matrix[i, :len(h)] = h
        if len(h) < max_len:
            matrix[i, len(h):] = h[-1] if h else 0.0
    return {'gens': np.arange(1, max_len + 1),
            'mean': np.mean(matrix, axis=0), 'std': np.std(matrix, axis=0)}


# ── M6: Pareto front size ─────────────────────────────────────────
def pareto_front_size(nsga2_results):
    """Mean ± std number of non-dominated solutions.

    Parameters:
        nsga2_results (list[dict]): Each with 'pareto_F'.
    Returns:
        dict: mean, std.
    """
    sizes = [r['pareto_F'].shape[0] for r in nsga2_results]
    return {'mean': float(np.mean(sizes)), 'std': float(np.std(sizes))}


# ── M7: GD and IGD ────────────────────────────────────────────────
def gd_igd_metrics(nsga2_results, ref_front):
    """Generational Distance and Inverted Generational Distance.

    Parameters:
        nsga2_results (list[dict]): Each with 'pareto_F'.
        ref_front (np.ndarray): Reference Pareto front.
    Returns:
        dict: 'gd_mean', 'gd_std', 'igd_mean', 'igd_std'.
    """
    gd_vals, igd_vals = [], []
    gd_ind = GD(ref_front)
    igd_ind = IGD(ref_front)
    for r in nsga2_results:
        F = r['pareto_F']
        if F.shape[0] > 0:
            gd_vals.append(gd_ind(F))
            igd_vals.append(igd_ind(F))
    gd_arr = np.array(gd_vals) if gd_vals else np.array([0.0])
    igd_arr = np.array(igd_vals) if igd_vals else np.array([0.0])
    return {'gd_mean': float(np.mean(gd_arr)), 'gd_std': float(np.std(gd_arr)),
            'igd_mean': float(np.mean(igd_arr)), 'igd_std': float(np.std(igd_arr))}


# ── Compute efficiency (for radar chart) ──────────────────────────
def compute_efficiency(evals_dict):
    """1/mean_evals_to_90pct, normalised 0–1 across algorithms.

    Parameters:
        evals_dict (dict): {algo_name: list_of_evals_to_90pct}.
    Returns:
        dict: {algo_name: normalised_efficiency}.
    """
    raw = {}
    for algo, vals in evals_dict.items():
        finite = [v for v in vals if np.isfinite(v)]
        raw[algo] = 1.0 / np.mean(finite) if finite else 0.0
    max_val = max(raw.values()) if raw else 1.0
    if max_val == 0:
        max_val = 1.0
    return {a: v / max_val for a, v in raw.items()}
