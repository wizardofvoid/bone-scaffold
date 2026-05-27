"""
Bone Scaffold Optimisation Problem definition.
Surrogate functions, constraints, bounds, and pymoo Problem class.
"""

import numpy as np

# Design variable bounds
BOUNDS = {
    'xl': np.array([0.2, 1.0, 0.5, 0.0]),
    'xu': np.array([2.0, 5.0, 0.90, 90.0])
}


def compressive_stiffness(x, rng=None):
    """
    Gibson-Ashby power law approximation for TPMS scaffold stiffness.

    Parameters:
        x (array-like): [wall_thickness, unit_cell_size, porosity_fraction, strut_angle].
        rng (np.random.Generator, optional): Seeded RNG for reproducible noise.

    Returns:
        float: Compressive stiffness in MPa (>= 0).
    """
    t, c, p, a = x[0], x[1], x[2], x[3]
    base = 1200 * ((1 - p) ** 1.8)
    thickness_factor = (t / c) ** 0.5
    angle_factor = 1.0 + 0.3 * np.sin(np.radians(a))
    noise = rng.normal(0, 15) if rng is not None else np.random.normal(0, 15)
    return max(0.0, base * thickness_factor * angle_factor + noise)


def porosity_score(x):
    """Direct porosity readout from chromosome.

    Parameters:
        x (array-like): Design variables.
    Returns:
        float: Porosity fraction.
    """
    return x[2]


def constraint_violations(x):
    """Compute constraint violations.

    C1: unit_cell_size >= 2 * wall_thickness
    C2: (wall_thickness / unit_cell_size) <= 0.45

    Returns:
        tuple: (viol_c1, viol_c2) each >= 0.
    """
    t, c = x[0], x[1]
    return max(0.0, 2 * t - c), max(0.0, (t / c) - 0.45)


def penalty(x):
    """Constraint penalty: 1e6 * sum-of-squared violations.

    Parameters:
        x (array-like): Design variables.
    Returns:
        float: Penalty value.
    """
    v1, v2 = constraint_violations(x)
    return 1e6 * (v1 ** 2 + v2 ** 2)


def is_feasible(x):
    """Check if design satisfies all constraints.

    Parameters:
        x (array-like): Design variables.
    Returns:
        bool: True if feasible.
    """
    v1, v2 = constraint_violations(x)
    return v1 <= 1e-9 and v2 <= 1e-9


# --------------- pymoo Problem ---------------
try:
    from pymoo.core.problem import Problem as _Problem
except ImportError:
    _Problem = object


class ScaffoldProblem(_Problem):
    """Multi-objective scaffold problem for pymoo NSGA-II.

    Objectives (minimise):
        f1 = -compressive_stiffness(x)
        f2 = -porosity_score(x)
    Constraints (g <= 0):
        g1 = 2*t - c
        g2 = t/c - 0.45

    Parameters:
        rng (np.random.Generator, optional): For reproducible stiffness noise.
    """

    def __init__(self, rng=None):
        self.rng = rng
        super().__init__(n_var=4, n_obj=2, n_constr=2,
                         xl=BOUNDS['xl'], xu=BOUNDS['xu'])

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population matrix x (pop_size, 4)."""
        n = x.shape[0]
        f1, f2, g1, g2 = [np.zeros(n) for _ in range(4)]
        for i in range(n):
            f1[i] = -compressive_stiffness(x[i], rng=self.rng)
            f2[i] = -porosity_score(x[i])
            g1[i] = 2.0 * x[i, 0] - x[i, 1]
            g2[i] = (x[i, 0] / x[i, 1]) - 0.45
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])
