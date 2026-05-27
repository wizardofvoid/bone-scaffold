"""
Bayesian Optimisation runner using scikit-optimize (gp_minimize).
Uses module-level _current_seed so bo_objective stays top-level and picklable.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from .problem import BOUNDS, compressive_stiffness, penalty, is_feasible, porosity_score

# Module-level variable for seed — avoids closures, keeps bo_objective picklable
_current_seed = None

# Search space
_SPACE = [
    Real(float(BOUNDS['xl'][0]), float(BOUNDS['xu'][0]), name='wall_thickness'),
    Real(float(BOUNDS['xl'][1]), float(BOUNDS['xu'][1]), name='unit_cell_size'),
    Real(float(BOUNDS['xl'][2]), float(BOUNDS['xu'][2]), name='porosity_fraction'),
    Real(float(BOUNDS['xl'][3]), float(BOUNDS['xu'][3]), name='strut_angle'),
]


def bo_objective(params):
    """Objective for gp_minimize. Top-level, no closure — picklable on Windows.

    Minimises: -stiffness + penalty  (equivalent to maximising stiffness subject to constraints).

    Parameters:
        params (list): [wall_thickness, unit_cell_size, porosity_fraction, strut_angle].
    Returns:
        float: Objective value to minimise.
    """
    rng = np.random.default_rng(_current_seed)
    x = np.array(params)
    return -compressive_stiffness(x, rng=rng) + penalty(x)


def run_bo(seed):
    """Run Bayesian Optimisation for one seed. Top-level for pickling.

    Parameters:
        seed (int): Random seed.
    Returns:
        dict: best_per_call, final_best_x, best_stiffness, best_porosity,
              is_feasible, seed, n_evals.
    """
    global _current_seed
    _current_seed = seed

    import os
    if os.environ.get("SCAFFOLD_USE_ANSYS") == "1":
        n_calls = 30
        n_random_starts = 5
    else:
        n_calls = 150
        n_random_starts = 20

    result = gp_minimize(
        bo_objective,
        _SPACE,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        acq_func='EI',
        noise=15.0 ** 2,
        random_state=seed,
        verbose=False,
    )

    # Build best-so-far curve
    best_so_far = []
    best_val = np.inf
    for v in result.func_vals:
        best_val = min(best_val, v)
        best_so_far.append(best_val)

    best_x = np.array(result.x)
    rng = np.random.default_rng(seed)
    stiff = compressive_stiffness(best_x, rng=rng)

    return {
        'best_per_call': best_so_far,
        'final_best_x': best_x,
        'best_stiffness': float(stiff),
        'best_porosity': float(porosity_score(best_x)),
        'is_feasible': bool(is_feasible(best_x)),
        'seed': seed,
        'n_evals': n_calls,
        'eval_history': [(i + 1, best_so_far[i]) for i in range(len(best_so_far))],
    }
