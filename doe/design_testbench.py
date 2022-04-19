import pandas as pd
import numpy as np
from doe.design import get_formula_from_string, constraints_as_scipy_constraints
import opti
from scipy.optimize import minimize
from typing import Union, Optional
from formulaic import Formula
import warnings
from typing import Callable
from numba import jit
from doe.JacobianForLogdet import JacobianForLogdet

@jit(nopython=True)
def logD(A: np.ndarray, delta: float = 1e-3) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    eigvals = np.linalg.eigvalsh(A.T@A + delta*np.eye(A.shape[1]))
    return np.sum(np.log(eigvals))

def get_objective(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    delta: float = 1e-3,
) -> Callable:
    """Returns a function that computes the combined objective.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.

    Returns:
        A function computing the weighted sum of the logD value and the constraint violation
        for a given input.

    """
    D = problem.n_inputs
    model_formula = get_formula_from_string(problem, model_type, rhs_only=True)

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD(A.to_numpy(), delta=delta)
        return obj

    return objective

#TODO: testen
def optimal_design(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given opti problem and model.
    
    Args:
        problem (opti.Problem): problem containing the inputs and constraints.
        model_type (str, Formula): keyword or formulaic Formula describing the model.
        n_experiments (int): Number of experiments. By default the value corresponds to
            the number of model terms + 3.
        tol (float): Tolerance for equality/NChooseK constraint violation. Default value is 1e-3.
        delta (float): Regularization parameter. Default value is 1e-3.

    
    
    """

    D = problem.n_inputs
    model_formula = get_formula_from_string(problem, model_type, rhs_only=True)

    # David fragen/selbst nachdenken: so in Ordnung?
    n_experiments_min = (
            len(model_formula.terms) + 2 #- n_ignore_eigvals(problem, model_formula) 
        )
    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}.")

    # get objective function
    objective = get_objective(problem, model_type, delta=delta)

    # get jacobian
    J = JacobianForLogdet(problem, model_formula, n_experiments)

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(problem, n_experiments, tol)

    # do the optimization
    result = minimize(
        objective,
        x0=problem.sample_inputs(n_experiments).values.reshape(-1),
        method= "SLSQP",
        bounds= [(p.bounds) for p in problem.inputs] * n_experiments,
        constraints= constraints,
        options= {"disp":True, "maxiter":2},
        jac = J.jacobian
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A


ndim = 17

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
X = optimal_design(problem, "fully-quadratic")
print(time.time()-t)
print(np.round(X,2))
