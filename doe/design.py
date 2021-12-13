from itertools import combinations

import numpy as np
import opti
import pandas as pd
from scipy.optimize import dual_annealing


def num_experiments(
    problem: opti.Problem, model_type: str = "linear", n_dof: int = 3
) -> int:
    """Determine the number of experiments needed to fit a model to a given problem.

    Args:
        problem (opti.Problem): Specification of the design and objective space.
        model_type (str, optional): Type of model. Defaults to "linear".
        n_dof (int, optional): Additional experiments to add degrees of freedom to a resulting model fit. Defaults to 3.

    Returns:
        int: Number of experiments.
    """
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
        n_experiments = n_linear + n_dof
    elif model_type == "linear-and-quadratic":
        n_experiments = n_linear + n_quadratic + n_dof
    elif model_type == "linear-and-interactions":
        n_experiments = n_linear + n_interaction + n_dof
    elif model_type == "fully-quadratic":
        n_experiments = n_linear + n_interaction + n_quadratic + n_dof
    else:
        raise Exception(f"Unknown model type {model_type}")
    return int(n_experiments)


def logD(A: np.ndarray) -> float:
    """log(|F|)"""
    F = A.T @ A
    _, logd = np.linalg.slogdet(F)
    return logd


def optimal_design(problem: opti.Problem, model_type="linear") -> pd.DataFrame:
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

    result = dual_annealing(
        objective,
        x0=problem.sample_inputs(N).values.reshape(-1),
        bounds=[(p.bounds) for p in problem.inputs] * N,
        maxiter=100,
        maxfun=10000,
    )

    A = result["x"].reshape(N, D)
    return pd.DataFrame(
        A, columns=problem.inputs.names, index=[f"exp{i}" for i in range(len(A))]
    )
