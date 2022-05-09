import warnings
from typing import Callable, Dict, Optional, Union

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
    metrics,
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
    ipopt_options: Dict = {},
    linearize_NChooseK: bool = False,
    jacobian_building_block: Callable = None,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given opti problem and model.

    Args:
        problem (opti.Problem): problem containing the inputs and constraints.
        model_type (str, Formula): keyword or formulaic Formula describing the model.
        n_experiments (int): Number of experiments. By default the value corresponds to
            the number of model terms + 3.
        tol (float): Tolerance for equality/NChooseK constraint violation. Default value is 1e-3.
        delta (float): Regularization parameter. Default value is 1e-3.
        ipopt_options (Dict): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
        linearize_NChooseK (bool): Tries to replace NChooseK constraints by linear constraints if set
            to True. For details see nchoosek_constraint_as_scipy_linear_constraint() function.
        jacobian_building_block (Callable): Only needed for models of higher order than 3. derivatives
            of each model term with respect to each input variable.

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
            if not isinstance(c, opti.NChooseK):
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
    J = JacobianForLogdet(
        problem,
        model_formula,
        n_experiments,
        delta=delta,
        jacobian_building_block=jacobian_building_block,
    )

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(
        problem, n_experiments, tol, linearize_NChooseK
    )

    # set ipopt options
    _ipopt_options = {"maxiter": 500, "disp": 0}
    for key in ipopt_options.keys():
        _ipopt_options[key] = ipopt_options[key]
    if _ipopt_options["disp"] > 12:
        _ipopt_options["disp"] = 0

    # do the optimization
    result = minimize_ipopt(
        objective,
        x0=x0,
        bounds=[(p.bounds) for p in problem.inputs] * n_experiments,
        # "SLSQP" has no deeper meaning here and just ensures correct constraint standardization
        constraints=standardize_constraints(constraints, x0, "SLSQP"),
        options=_ipopt_options,
        jac=J.jacobian,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D),
        columns=problem.inputs.names,
        index=[f"exp{i}" for i in range(n_experiments)],
    )

    # exit message
    if _ipopt_options[b"print_level"] > 12:
        for key in ["fun", "message", "nfev", "nit", "njev", "status", "success"]:
            print(key + ":", result[key])
        X = model_formula.get_model_matrix(A).to_numpy()
        d = metrics(X, problem, n_samples=1000)
        print("metrics:", d)

    return A
