import warnings
from typing import Callable, Optional, Union

import numpy as np
import opti
import pandas as pd
from cyipopt import minimize_ipopt
from formulaic import Formula
from scipy.optimize._minimize import standardize_constraints

from doe.JacobianForLogdet import JacobianForLogdet
from doe.utils import (
    constraints_as_scipy_constraints,
    get_formula_from_string,
    n_zero_eigvals,
)


def logD(A: np.ndarray, delta: float = 1e-7) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    return np.linalg.slogdet(A.T @ A + delta * np.eye(A.shape[1]))[1]


def get_objective(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    delta: float = 1e-7,
) -> Callable:
    """Returns a function that computes the objective value.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        delta (float): Regularization parameter for information matrix. Default value is 1e-3.

    Returns:
        A function computing the objective -logD for a given input vector x

    """
    D = problem.n_inputs
    model_formula = get_formula_from_string(
        problem=problem, model_type=model_type, rhs_only=True
    )

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD(A.to_numpy(), delta=delta)
        return obj

    return objective


def find_local_max_ipopt(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    tol: float = 1e-3,
    delta: float = 1e-7,
    disp: int = 0,
    maxiter: int = 500,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given opti problem and model.

    Args:
        problem (opti.Problem): problem containing the inputs and constraints.
        model_type (str, Formula): keyword or formulaic Formula describing the model.
        n_experiments (int): Number of experiments. By default the value corresponds to
            the number of model terms + 3.
        tol (float): Tolerance for equality/NChooseK constraint violation. Default value is 1e-3.
        delta (float): Regularization parameter. Default value is 1e-3.
        disp (int): Verbosity parameter for IPOPT. Valid range is 0 <= disp <= 12. Default value is 0.
        maxiter (int): maximum number of iterations. Default value is 100.

    Returns:
        A pd.DataFrame object containing the best found input for the experiments. This is only a
        local optimum.

    """

    D = problem.n_inputs
    model_formula = get_formula_from_string(
        problem=problem, model_type=model_type, rhs_only=True
    )

    # initial values and required number of experiments
    try:
        n_experiments_min = (
            len(model_formula.terms) + 3 - n_zero_eigvals(problem, model_formula)
        )
        if n_experiments is None:
            n_experiments = n_experiments_min
        elif n_experiments < n_experiments_min:
            warnings.warn(
                f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
            )
        x0 = problem.sample_inputs(n_experiments).values.reshape(-1)

    except Exception:
        # in case of exceptions only consider linear constraints
        warnings.warn(
            "Sampling of points fulfilling this problem's constraints is not implemented."
        )

        _constraints = []
        for c in problem.constraints:
            if isinstance(c, opti.LinearEquality) or isinstance(
                c, opti.LinearInequality
            ):
                _constraints.append(c)
        _problem = opti.Problem(
            inputs=problem.inputs, outputs=problem.outputs, constraints=_constraints
        )

        n_experiments_min = (
            len(model_formula.terms) + 3 - n_zero_eigvals(_problem, model_formula)
        )
        if n_experiments is None:
            n_experiments = n_experiments_min
        elif n_experiments < n_experiments_min:
            warnings.warn(
                f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
            )
        x0 = _problem.sample_inputs(n_experiments).values.reshape(-1)

    # get objective function
    objective = get_objective(problem, model_type, delta=delta)

    # get jacobian
    J = JacobianForLogdet(problem, model_formula, n_experiments, delta=delta)

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(problem, n_experiments, tol)

    # do the optimization
    result = minimize_ipopt(
        objective,
        x0=x0,
        bounds=[(p.bounds) for p in problem.inputs] * n_experiments,
        # "SLSQP" has no deeper meaning here and just ensures correct constraint standardization
        constraints=standardize_constraints(constraints, x0, "SLSQP"),
        options={"maxiter": maxiter, "disp": disp},
        jac=J.jacobian,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A
