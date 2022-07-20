import warnings
from typing import Callable, Dict, Optional, Union

import numpy as np
import opti
import pandas as pd
from cyipopt import minimize_ipopt
from formulaic import Formula
from scipy.optimize._minimize import standardize_constraints

from doe.jacobian import JacobianForLogdet
from doe.sampling import OptiSampling, Sampling
from doe.utils import (
    ProblemProvider,
    constraints_as_scipy_constraints,
    metrics,
    nchoosek_constraints_as_bounds,
)


def logD(A: np.ndarray, delta: float = 1e-7) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    return np.linalg.slogdet(A.T @ A + delta * np.eye(A.shape[1]))[1]


def get_objective(
    problem_provider: ProblemProvider,
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
    D = problem_provider.problem.n_inputs
    model_formula = problem_provider.get_formula_from_string(
        model_type=model_type, rhs_only=True
    )

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(
            x.reshape(len(x) // D, D), columns=problem_provider.problem.inputs.names
        )
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD(A.to_numpy(), delta=delta)
        return obj

    return objective


def find_local_max_ipopt(
    problem_provider: ProblemProvider,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    tol: float = 0,
    delta: float = 1e-7,
    ipopt_options: Dict = {},
    jacobian_building_block: Optional[Callable] = None,
    sampling: Union[Sampling, np.ndarray] = OptiSampling,
    fixed_experiments: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given opti problem and model.

    Args:
        problem (opti.Problem): problem containing the inputs and constraints.
        model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
            are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
        n_experiments (int): Number of experiments. By default the value corresponds to
            the number of model terms - dimension of ker() + 3.
        tol (float): Tolerance for linear/NChooseK constraint violation. Default value is 0.
        delta (float): Regularization parameter. Default value is 1e-3.
        ipopt_options (Dict): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
        jacobian_building_block (Callable): Only needed for models of higher order than 3. derivatives
            of each model term with respect to each input variable.
        sampling (Sampling, np.ndarray): Sampling class or a np.ndarray object containing the initial guess.
        fixed_experiments (np.ndarray): numpy array containing experiments that will definitely part of the design.
            Values are set before the optimization.

    Returns:
        A pd.DataFrame object containing the best found input for the experiments. This is only a
        local optimum.

    """
    D = problem_provider.problem.n_inputs
    model_formula = problem_provider.get_formula_from_string(
        model_type=model_type, rhs_only=True
    )

    # check if there are NChooseK constraints that must be ignored when sampling with opti.Problem.sample_inputs
    _problem = problem_provider.problem
    if problem_provider.problem.n_constraints > 0:
        if any(
            [isinstance(c, opti.NChooseK) for c in problem_provider.problem.constraints]
        ) and not all(
            [isinstance(c, opti.NChooseK) for c in problem_provider.problem.constraints]
        ):
            warnings.warn(
                "Sampling of points fulfilling this problem's constraints is not implemented."
            )

            _constraints = []
            for c in problem_provider.problem.constraints:
                if not isinstance(c, opti.NChooseK):
                    _constraints.append(c)
            _problem = opti.Problem(
                inputs=problem_provider.problem.inputs,
                outputs=problem_provider.problem.outputs,
                constraints=_constraints,
            )

    # determine number of experiments
    n_experiments_min = len(model_formula.terms) + 3
    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(
            f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
        )

    # initital values
    if isinstance(sampling, np.ndarray):
        x0 = sampling
    else:
        x0 = sampling(_problem).sample(n_experiments)

    # get objective function
    objective = get_objective(problem_provider, model_type, delta=delta)

    # get jacobian
    J = JacobianForLogdet(
        problem_provider.problem,
        model_formula,
        n_experiments,
        delta=delta,
        jacobian_building_block=jacobian_building_block,
    )

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(
        problem_provider.problem, n_experiments, tol
    )

    # find bounds imposing NChooseK constraints
    bounds = nchoosek_constraints_as_bounds(problem_provider.problem, n_experiments)

    # fix experiments if any are given
    if fixed_experiments is not None:
        fixed_experiments = np.array(fixed_experiments)
        check_fixed_experiments(
            problem_provider.problem, n_experiments, fixed_experiments
        )
        for i, val in enumerate(fixed_experiments.flatten()):
            bounds[i] = (val, val)
            x0[i] = val

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
        bounds=bounds,
        # "SLSQP" has no deeper meaning here and just ensures correct constraint standardization
        constraints=standardize_constraints(constraints, x0, "SLSQP"),
        options=_ipopt_options,
        jac=J.jacobian,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D),
        columns=problem_provider.problem.inputs.names,
        index=[f"exp{i}" for i in range(n_experiments)],
    )

    # project categorical variables back to categorical values
    for input in problem_provider.original_problem.inputs:
        if isinstance(input, opti.Categorical):
            cat_col = [
                input.domain[np.argmax(x.values)]
                for _, x in A[problem_provider._cat_dict[input.name]].iterrows()
            ]
            A[input.name] = cat_col
            A = A.drop(problem_provider._cat_dict[input.name], axis=1)

    # exit message
    if _ipopt_options[b"print_level"] > 12:
        for key in ["fun", "message", "nfev", "nit", "njev", "status", "success"]:
            print(key + ":", result[key])
        X = model_formula.get_model_matrix(A).to_numpy()
        d = metrics(X, problem_provider.problem, n_samples=1000)
        print("metrics:", d)

    return A


def check_fixed_experiments(
    problem: opti.Problem, n_experiments: int, fixed_experiments: np.ndarray
) -> None:
    """Checks if the shape of the fixed experiments is correct and if the number of fixed experiments is valid
    Args:
        problem (opti.Problem): problem defining the input variables used for the check.
        n_experiments (int): total number of experiments in the design that fixed_experiments are part of.
        fixed_experiments (np.ndarray): fixed experiment proposals to be checked.
    """

    n_fixed_experiments, D = np.array(fixed_experiments).shape

    if n_fixed_experiments >= n_experiments:
        raise ValueError(
            "For starting the optimization the total number of experiments must be larger that the number of fixed experiments."
        )

    if D != problem.n_inputs:
        raise ValueError(
            f"Invalid shape of fixed_experiments. Length along axis 1 is {D}, but must be {problem.n_inputs}"
        )
