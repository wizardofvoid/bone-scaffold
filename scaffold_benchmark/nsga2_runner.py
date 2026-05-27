"""
NSGA-II runner using pymoo.
Top-level functions only — no lambdas — for Windows pickling safety.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume

from .problem import ScaffoldProblem, BOUNDS, compressive_stiffness, porosity_score, is_feasible


class _HVCallback(Callback):
    """Records hypervolume at each generation."""

    def __init__(self, ref_point):
        super().__init__()
        self.ref_point = ref_point
        self.hv_history = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if self.ref_point is not None and len(F) > 0:
            hv = Hypervolume(ref_point=self.ref_point).do(F)
        else:
            hv = 0.0
        self.hv_history.append(hv)


def run_nsga2(seed, n_gen=150, ref_point=None):
    """Run NSGA-II for one seed. Top-level for pickling.

    Parameters:
        seed (int): Random seed.
        n_gen (int): Number of generations.
        ref_point (np.ndarray|None): HV reference point.
    Returns:
        dict: pareto_F, pareto_X, hv_history, seed, n_evals, pop_F, pop_X.
    """
    rng = np.random.default_rng(seed)
    problem = ScaffoldProblem(rng=rng)

    algorithm = NSGA2(pop_size=100)
    cb = _HVCallback(ref_point)

    res = pymoo_minimize(
        problem, algorithm,
        termination=('n_gen', n_gen),
        seed=seed,
        callback=cb,
        verbose=False,
    )

    pareto_F = res.F if res.F is not None else np.empty((0, 2))
    pareto_X = res.X if res.X is not None else np.empty((0, 4))

    # Best stiffness = max of -f1 (f1 is negated stiffness)
    if pareto_F.shape[0] > 0:
        best_stiffness = float(-pareto_F[:, 0].min())
    else:
        best_stiffness = 0.0

    return {
        'pareto_F': pareto_F,
        'pareto_X': pareto_X,
        'hv_history': cb.hv_history,
        'seed': seed,
        'n_evals': res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else n_gen * 100,
        'best_stiffness': best_stiffness,
    }


def run_nsga2_reference(seed=42, n_gen=500):
    """Run a long NSGA-II to produce a reference Pareto front.

    Parameters:
        seed (int): Random seed.
        n_gen (int): Generations for reference run.
    Returns:
        np.ndarray: Reference Pareto front F (n_points, 2).
    """
    rng = np.random.default_rng(seed)
    problem = ScaffoldProblem(rng=rng)
    algorithm = NSGA2(pop_size=100)

    res = pymoo_minimize(
        problem, algorithm,
        termination=('n_gen', n_gen),
        seed=seed,
        verbose=False,
    )
    return res.F if res.F is not None else np.empty((0, 2))
