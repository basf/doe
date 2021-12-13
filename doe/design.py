from itertools import combinations

import numpy as np
import opti
import pandas as pd
from scipy.optimize import dual_annealing


def num_experiments(problem, model_type="linear", n_dof=3):
    """Determine the number of experiments needed to fit a model to a given problem."""
    continuous = [p for p in problem.inputs if isinstance(p, opti.Continuous)]
    discretes = [p for p in problem.inputs if isinstance(p, opti.Discrete)]
    categoricals = [p for p in problem.inputs if isinstance(p, opti.Categorical)]

    constraints = problem.constraints
    if constraints is None:
        constraints = []
    equalities = [c for c in constraints if isinstance(c, opti.LinearEquality)]

    # number of linearly independent numeric variables
    d = len(continuous) + len(discretes) - len(equalities)

    # categoricals are encoded using dummy variables, hence a categorical with 4 levels requires 3 experiments in a linear model.
    dummy_levels = [len(p.domain) - 1 for p in categoricals]

    n_linear = 1 + d + sum(dummy_levels)
    n_quadratic = d  # dummy variables are not squared
    n_interaction = d * (d - 1) / 2
    n_interaction += d * sum(dummy_levels)
    n_interaction += sum([np.prod(k) for k in combinations(dummy_levels, 2)])

    if model_type == "linear":
        return n_linear
    elif model_type == "linear-and-quadratic":
        return n_linear + n_quadratic
    elif model_type == "linear-and-interactions":
        return n_linear + n_interaction
    elif model_type == "fully-quadratic":
        return n_linear + n_interaction + n_quadratic


def logD(A):
    """log(|F|)"""
    F = A.T @ A
    _, logd = np.linalg.slogdet(F)
    return logd


def optimal_design(problem, model_type="linear"):
    """Generate a D-optimal design for a given problem assuming a linear model."""
    D = problem.n_inputs
    N = num_experiments(problem, model_type)
    if model_type != "linear":
        raise NotImplementedError("Currently, only linear models implemented.")

    def objective(x):
        A = x.reshape(N, D)
        obj = -logD(A)
        if problem.constraints is not None:
            X = pd.DataFrame(A, columns=problem.inputs.names)
            penalty = problem.constraints(X).clip(0, None).sum().values[0]
            obj += penalty
        return obj

    A0 = problem.sample_inputs(N).values  # initial guess
    bounds = [(p.bounds) for p in problem.inputs] * N  # box bounds

    res = dual_annealing(objective, x0=A0.reshape(-1), bounds=bounds, maxiter=100)

    A = res["x"].reshape(N, D)
    return A
