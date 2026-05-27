"""
Bone Scaffold Optimisation Problem definition.
Surrogate functions, constraints, bounds, and pymoo Problem class.
"""

import numpy as np
import os
import math

# Design variable bounds
BOUNDS = {
    'xl': np.array([0.2, 1.0, 0.5, 0.0]),
    'xu': np.array([2.0, 5.0, 0.90, 90.0])
}

# Module-level variables for ANSYS availability tracking
_ANSYS_AVAILABLE = True
_ANSYS_WARNING_SHOWN = False


def ansys_compressive_stiffness(x):
    """
    Runs Finite Element Analysis (FEA) on a single unit cell of the scaffold using PyMAPDL.
    Calculates equivalent compressive stiffness in MPa.
    """
    global _ANSYS_WARNING_SHOWN
    try:
        from ansys.mapdl.core import launch_mapdl
    except ImportError as e:
        if not _ANSYS_WARNING_SHOWN:
            print("\n[WARNING] ansys-mapdl-core is not installed. Falling back to Surrogate Model.")
            _ANSYS_WARNING_SHOWN = True
        raise e

    mapdl = None
    try:
        # Launch MAPDL in batch mode with a quick warning-only log level
        mapdl = launch_mapdl(loglevel="WARNING")
        mapdl.clear()
        
        # Extract variables
        t, c, p, a = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        
        # Calculate pore radius from target porosity: V_pore = p * c^3 = 4/3 * pi * r_pore^3
        r_pore = c * ((3.0 * p) / (4.0 * math.pi)) ** (1.0 / 3.0)
        
        # Calculate equivalent elastic modulus of bulk cell walls (in MPa)
        E_base = 3000.0 * ((t / c) ** 0.5) * (1.0 + 0.3 * math.sin(math.radians(a)))
        
        # 1. Enter preprocessor
        mapdl.prep7()
        
        # 2. Define Material Properties
        mapdl.et(1, "SOLID186")       # 20-node structural solid
        mapdl.mp("EX", 1, E_base)     # Young's Modulus in MPa
        mapdl.mp("NUXY", 1, 0.3)      # Poisson's ratio
        
        # 3. Create Solid Geometry (Unit cell block with spherical pore)
        v_block = mapdl.block(0, c, 0, c, 0, c)
        v_sphere = mapdl.sph4(c / 2.0, c / 2.0, c / 2.0, r_pore)
        
        # Subtract sphere from block (Volume 1 minus Volume 2)
        mapdl.vsbv(v_block, v_sphere)
        
        # 4. Meshing (coarse mesh size to ensure super-fast solver run times)
        mapdl.esize(c / 8.0)
        mapdl.vmesh("ALL")
        
        # 5. Boundary Conditions (Compression test)
        # Fix bottom surface nodes (Z = 0)
        mapdl.nsel("S", "LOC", "Z", 0.0)
        mapdl.d("ALL", "ALL", 0.0)
        
        # Apply 1% displacement at top surface nodes (Z = c)
        disp = -0.01 * c
        mapdl.nsel("S", "LOC", "Z", c)
        mapdl.d("ALL", "UZ", disp)
        
        mapdl.allsel()
        
        # 6. Solve static analysis
        mapdl.run("/SOLU")
        mapdl.antype("STATIC")
        mapdl.solve()
        
        # 7. Post-Processing: Sum reaction forces at the bottom nodes
        mapdl.post1()
        mapdl.nsel("S", "LOC", "Z", 0.0)
        mapdl.fsum()
        reaction_force = mapdl.get(0, "FSUM", 0, "FTOT", "Z")
        
        # Equivalent compressive stiffness (Elastic Modulus in MPa) E_eff = K * L / A = K * c / c^2 = K / c
        stiffness = abs(reaction_force / disp) / c
        
        # Clean MAPDL shutdown
        mapdl.exit()
        return float(stiffness)

    except Exception as e:
        if mapdl is not None:
            try:
                mapdl.exit()
            except:
                pass
        if not _ANSYS_WARNING_SHOWN:
            print(f"\n[WARNING] Could not run ANSYS FEA: {e}. Falling back to Surrogate Model.")
            _ANSYS_WARNING_SHOWN = True
        raise e


def compressive_stiffness(x, rng=None):
    """
    Calculates compressive stiffness.
    Uses ANSYS FEA if os.environ["SCAFFOLD_USE_ANSYS"] == "1", with a graceful fallback to the
    Gibson-Ashby surrogate model.
    """
    global _ANSYS_AVAILABLE
    if _ANSYS_AVAILABLE and os.environ.get("SCAFFOLD_USE_ANSYS") == "1":
        try:
            return ansys_compressive_stiffness(x)
        except Exception:
            _ANSYS_AVAILABLE = False  # Mark unavailable to skip future calls and run fast!

    # --- Gibson-Ashby Surrogate Model ---
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
