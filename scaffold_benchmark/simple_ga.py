"""
Simple Genetic Algorithm — pure NumPy, no external GA library.
Binary tournament, SBX crossover, polynomial mutation, elitism.
"""

import numpy as np
from .problem import BOUNDS, compressive_stiffness, porosity_score, penalty, is_feasible


def sbx_crossover(p1, p2, xl, xu, eta=20, prob=0.9, rng=None):
    """Simulated Binary Crossover (SBX).

    Parameters:
        p1, p2 (np.ndarray): Parent chromosomes.
        xl, xu (np.ndarray): Variable bounds.
        eta (float): Distribution index.
        prob (float): Crossover probability.
        rng (np.random.Generator): RNG.
    Returns:
        tuple: Two offspring arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > prob:
        return p1.copy(), p2.copy()

    c1, c2 = np.empty_like(p1), np.empty_like(p2)
    for i in range(len(p1)):
        if rng.random() < 0.5:
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            if abs(y1 - y2) > 1e-9:
                u = rng.random()
                if u <= 0.5:
                    beta_q = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1[i] = np.clip(0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2), xl[i], xu[i])
                c2[i] = np.clip(0.5 * ((1 - beta_q) * y1 + (1 + beta_q) * y2), xl[i], xu[i])
            else:
                c1[i], c2[i] = p1[i], p2[i]
        else:
            c1[i], c2[i] = p1[i], p2[i]
    return c1, c2


def polynomial_mutation(x, xl, xu, eta=20, prob=0.1, rng=None):
    """Polynomial Mutation.

    Parameters:
        x (np.ndarray): Chromosome.
        xl, xu (np.ndarray): Variable bounds.
        eta (float): Distribution index.
        prob (float): Per-variable mutation probability.
        rng (np.random.Generator): RNG.
    Returns:
        np.ndarray: Mutated chromosome.
    """
    if rng is None:
        rng = np.random.default_rng()
    xm = x.copy()
    for i in range(len(x)):
        if rng.random() < prob:
            y, yl, yu = x[i], xl[i], xu[i]
            if yu - yl > 1e-9:
                d1 = (y - yl) / (yu - yl)
                d2 = (yu - y) / (yu - yl)
                u = rng.random()
                if u <= 0.5:
                    val = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - d1) ** (eta + 1.0))
                    dq = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - d2) ** (eta + 1.0))
                    dq = 1.0 - val ** (1.0 / (eta + 1.0))
                xm[i] = np.clip(y + dq * (yu - yl), yl, yu)
    return xm


def tournament_select(pop, fitnesses, tournament_size=2, rng=None):
    """Binary tournament selection.

    Parameters:
        pop (np.ndarray): Population (pop_size, n_var).
        fitnesses (np.ndarray): Fitness values.
        tournament_size (int): Tournament size.
        rng (np.random.Generator): RNG.
    Returns:
        np.ndarray: Selected chromosome copy.
    """
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(pop.shape[0], size=tournament_size, replace=False)
    return pop[idx[np.argmax(fitnesses[idx])]].copy()


def run_simple_ga(seed):
    """Run Simple GA for one seed. Top-level function for Windows pickling.

    Parameters:
        seed (int): Random seed.
    Returns:
        dict: Keys: best_per_gen, eval_history, final_best, best_stiffness,
              best_porosity, is_feasible, final_pop, stiffness_history, seed.
    """
    rng = np.random.default_rng(seed)
    import os
    if os.environ.get("SCAFFOLD_USE_ANSYS") == "1":
        pop_size, n_gen, elite = 20, 10, 2
    else:
        pop_size, n_gen, elite = 100, 150, 5
    xl, xu = BOUNDS['xl'], BOUNDS['xu']

    pop = rng.uniform(xl, xu, size=(pop_size, 4))
    eval_count = 0
    best_fit = -np.inf
    best_x = None
    eval_hist = []

    def _eval(x):
        nonlocal eval_count, best_fit, best_x
        s = compressive_stiffness(x, rng=rng)
        f = s - penalty(x)
        eval_count += 1
        if f > best_fit:
            best_fit = f
            best_x = x.copy()
        eval_hist.append((eval_count, best_fit))
        return f, s

    fits = np.zeros(pop_size)
    stiffs = np.zeros(pop_size)
    for i in range(pop_size):
        fits[i], stiffs[i] = _eval(pop[i])

    best_per_gen = [float(np.max(fits))]
    stiff_hist = [float(stiffs[np.argmax(fits)])]

    for _ in range(n_gen):
        order = np.argsort(fits)[::-1]
        pop, fits, stiffs = pop[order], fits[order], stiffs[order]

        nxt = np.zeros_like(pop)
        nf, ns = np.zeros(pop_size), np.zeros(pop_size)
        nxt[:elite] = pop[:elite]
        nf[:elite], ns[:elite] = fits[:elite], stiffs[:elite]

        idx = elite
        while idx < pop_size:
            p1 = tournament_select(pop, fits, 2, rng)
            p2 = tournament_select(pop, fits, 2, rng)
            c1, c2 = sbx_crossover(p1, p2, xl, xu, 20, 0.9, rng)
            c1 = polynomial_mutation(c1, xl, xu, 20, 0.1, rng)
            c2 = polynomial_mutation(c2, xl, xu, 20, 0.1, rng)
            if idx < pop_size:
                nxt[idx] = c1
                nf[idx], ns[idx] = _eval(c1)
                idx += 1
            if idx < pop_size:
                nxt[idx] = c2
                nf[idx], ns[idx] = _eval(c2)
                idx += 1

        pop, fits, stiffs = nxt, nf, ns
        best_per_gen.append(float(np.max(fits)))
        stiff_hist.append(float(stiffs[np.argmax(fits)]))

    bi = np.argmax(fits)
    return {
        'best_per_gen': best_per_gen,
        'eval_history': eval_hist,
        'final_best': pop[bi],
        'best_stiffness': float(compressive_stiffness(pop[bi], rng=rng)),
        'best_porosity': float(porosity_score(pop[bi])),
        'is_feasible': bool(is_feasible(pop[bi])),
        'final_pop': pop,
        'stiffness_history': stiff_hist,
        'seed': seed,
    }
