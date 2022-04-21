import warnings
from typing import Callable, Optional, Union, Dict

import numpy as np
import opti
import pandas as pd
from cyipopt import minimize_ipopt
from formulaic import Formula
from numba import jit
from scipy.optimize._minimize import standardize_constraints
from scipy.optimize import minimize

from doe.design import constraints_as_scipy_constraints, get_formula_from_string, n_ignore_eigvals, get_callback
from doe.JacobianForLogdet import JacobianForLogdet
from doe.basinhopping_ipopt import basinhopping_ipopt


#TODO: testen
#TODO: ggf durch np.linalg.det oder np.linalg.slogdet ersetzen --> bessere laufzeit ab ca. n=1000
@jit(nopython=True)
def logD(A: np.ndarray, delta: float = 1e-3) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    eigvals = np.linalg.eigvalsh(A.T @ A + delta * np.eye(A.shape[1]))
    return np.sum(np.log(eigvals))

#TODO: testen
def get_objective(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    delta: float = 1e-3,
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
    model_formula = get_formula_from_string(problem=problem, model_type=model_type, rhs_only=True)

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD(A.to_numpy(), delta=delta)
        return obj

    return objective


# TODO: testen
def find_local_max_ipopt(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    tol: float = 1e-3,
    delta: float = 1e-3,
    disp: int = 0,
    maxiter: int = 100,
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
    model_formula = get_formula_from_string(problem=problem, model_type=model_type, rhs_only=True)

    #compute required number of experiments
    n_experiments_min = (
        len(model_formula.terms) + 3 - n_ignore_eigvals(problem, model_formula)
    )
    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(
            f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
        )

    # get objective function
    objective = get_objective(problem, model_type, delta=delta)

    # get jacobian
    J = JacobianForLogdet(problem, model_formula, n_experiments, delta=delta)

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(problem, n_experiments, tol)

    # method used
    # TODO: Eigentlich überflüssig hier, oder? --> rausnehmen
    method = "SLSQP"

    # initial values
    x0 = problem.sample_inputs(n_experiments).values.reshape(-1)

    # do the optimization
    result = minimize_ipopt(
        objective,
        x0=x0,
        method=method,
        bounds=[(p.bounds) for p in problem.inputs] * n_experiments,
        constraints=standardize_constraints(constraints, x0, method),
        options={"maxiter": maxiter, "disp": disp},
        jac=J.jacobian,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A


#TODO: testen
def optimal_design(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    n_max_min_found: int = 1e2,
    n_max_no_change: int = 10,
    tol: float = 1e-3,
    delta: float = 1e-3,
    minimizer_kwargs: Dict = {},
    jacobian_building_block: Optional[Callable] =None,
    verbose=False,
) -> pd.DataFrame:
    """Generate a D-optimal design for a given problem assuming a linear model.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        n_experiment (int): Optional argument to explicitly set the number of experiments.
            Default value is None.
        n_max_min_found (int): Algorithm stops after this number of found minima. Default value is 100
        n_max_no_change (int): Algorithm stops if the last n_max_no_change minima have
            not improved the best discovered minimum. Default value is 10.
        tol (float): tolarance for the computation of constraint violation. Default value is 1e-3.
        delta (float): Regularization parameter. Default value is 1e-3.
        verbose (bool): status messages are shown regularily if set to True.
        minimizer_kwargs (Dict): A dictionary containing kwargs for minimize_ipopt.
        jacobian_building_block (Callable): Optional parameter. A callable that computes the building block of the Jacobian of the objective.

    Returns:
        A pd.DataFrame object containing the best found input values for the experiments.
    """
    # TODO: unterstützung für categorical inputs
    D = problem.n_inputs
    model_formula = get_formula_from_string(model_type=model_type, problem=problem, rhs_only=True)

    #compute required number of experiments
    n_experiments_min = (
        len(model_formula.terms) + 3 - n_ignore_eigvals(problem, model_formula)
    )
    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(
            f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
        )

    # get objective function
    objective = get_objective(problem, model_type, delta=delta)

    # get jacobian
    J = JacobianForLogdet(problem, model_formula, n_experiments, delta=delta, jacobian_building_block=jacobian_building_block)

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(problem, n_experiments, tol)

    # method used
    method = "SLSQP"

    # get callback for stop criterion an status messages
    callback = get_callback(
        problem=problem,
        n_max_min_found=n_max_min_found,
        n_max_no_change=n_max_no_change,
        tol=tol,
        verbose=verbose,
    )

    # write accept_test function to test if a step is inside the bounds
    def accept_test(**kwargs):
        x = kwargs["x_new"]
        test_max = bool(np.all(x <= accept_test.ub))
        test_min = bool(np.all(x >= accept_test.lb))
        return test_max and test_min

    bounds = np.array([(p.bounds) for p in problem.inputs] * n_experiments).T
    accept_test.lb = bounds[0]
    accept_test.ub = bounds[1]

    # initial values
    x0 = problem.sample_inputs(n_experiments).values.reshape(-1)

    #prepare minimizer kwargs
    _minimizer_kwargs = {
        "method": method,
        "options": {"maxiter": 100, "disp":0},
    }

    for k in minimizer_kwargs.keys():
        _minimizer_kwargs[k] = minimizer_kwargs[k]

    _minimizer_kwargs["jac"] = J.jacobian
    _minimizer_kwargs["constraints"] = standardize_constraints(constraints, x0, method)
    _minimizer_kwargs["bounds"] = [(p.bounds) for p in problem.inputs] * n_experiments


    # do the optimization
    result = basinhopping_ipopt(
        objective,
        x0=x0,
        minimizer_kwargs=_minimizer_kwargs,
        callback=callback,
        accept_test=accept_test,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A


ndim = 15

problem = opti.Problem(
    inputs = opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(ndim)]),
    outputs = [opti.Continuous("y")],
    constraints = [opti.LinearEquality(names=[f"x{i+1}" for i in range(ndim)], rhs=1)] 
#    + [opti.LinearInequality(names=[f"x{i+1}"], lhs=[1], rhs=1) for i in range(ndim)]
#    + [opti.LinearInequality(names=[f"x{i+1}"], lhs=[-1], rhs=0) for i in range(ndim)]
)


import time
#np.random.seed(1)

t = time.time()
X = find_local_max_ipopt(problem, "fully-quadratic", disp=5, maxiter=500)
print(time.time()-t)
print(np.round(X,2))
