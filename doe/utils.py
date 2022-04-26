from typing import Optional, Union

import numpy as np
import opti
import pandas as pd
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint


def get_formula_from_string(
    model_type: Union[str, Formula] = "linear",
    problem: Optional[opti.Problem] = None,
    rhs_only: bool = True,
) -> Formula:
    """Reformulates a string describing a model or certain keywords as Formula objects.

    Args:
        model_type (str or Formula): A formula containing all model terms.
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
            Only needed if the model is defined by a keyword
        rhs_only (bool): The function returns only the right hand side of the formula if set to True.

    Returns:
        A Formula object describing the model that was given as string or keyword.
    """
    if isinstance(model_type, Formula):
        return model_type

    # build model if a keyword and a problem are given.
    else:
        # linear model
        if model_type == "linear":
            assert (
                problem is not None
            ), "If the model is described by a keyword a problem must be provided"
            formula = "".join([input.name + " + " for input in problem.inputs])

        # linear and interactions model
        elif model_type == "linear-and-quadratic":
            assert (
                problem is not None
            ), "If the model is described by a keyword a problem must be provided."
            formula = "".join([input.name + " + " for input in problem.inputs])
            formula += "".join(
                ["{" + input.name + "**2} + " for input in problem.inputs]
            )

        # linear and quadratic model
        elif model_type == "linear-and-interactions":
            assert (
                problem is not None
            ), "If the model is described by a keyword a problem must be provided."
            formula = "".join([input.name + " + " for input in problem.inputs])
            for i in range(problem.n_inputs):
                for j in range(i):
                    formula += (
                        problem.inputs.names[j] + ":" + problem.inputs.names[i] + " + "
                    )

        # fully quadratic model
        elif model_type == "fully-quadratic":
            assert (
                problem is not None
            ), "If the model is described by a keyword a problem must be provided."
            formula = "".join([input.name + " + " for input in problem.inputs])
            for i in range(problem.n_inputs):
                for j in range(i):
                    formula += (
                        problem.inputs.names[j] + ":" + problem.inputs.names[i] + " + "
                    )
            formula += "".join(
                ["{" + input.name + "**2} + " for input in problem.inputs]
            )

        else:
            formula = model_type + "   "

    formula = Formula(formula[:-3])

    if rhs_only:
        if hasattr(formula, "rhs"):
            formula = formula.rhs

    return formula


def n_zero_eigvals(
    problem: opti.Problem, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Computes the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""
    # sample points (fulfilling the constraints)
    model_formula = get_formula_from_string(
        model_type=model_type, problem=problem, rhs_only=True
    )
    N = len(model_formula.terms) + 3
    A = problem.sample_inputs(N)

    # compute eigenvalues of information matrix
    model_matrix = model_formula.get_model_matrix(A)
    eigvals = np.abs(np.linalg.eigvalsh(model_matrix.T @ model_matrix))

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def constraints_as_scipy_constraints(
    problem: opti.Problem,
    n_experiments: int,
    tol: float = 1e-3,
):
    """Formulates opti constraints as scipy constraints.

    Args:
        problem (opti.Problem): problem whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.
    """
    D = problem.n_inputs

    constraints = []
    if problem.constraints is None:
        return constraints
    for c in problem.constraints:
        if isinstance(c, opti.LinearEquality):
            # write lower/upper bound as vector
            lb = np.ones(n_experiments) * (c.rhs - tol)
            ub = np.ones(n_experiments) * (c.rhs + tol)

            # write constraint as matrix
            lhs = {c.names[i]: c.lhs[i] for i in range(len(c.names))}
            row = np.zeros(D)
            for i, name in enumerate(problem.inputs.names):
                if name in lhs.keys():
                    row[i] = lhs[name]

            A = np.zeros(shape=(n_experiments, D * n_experiments))
            for i in range(n_experiments):
                A[i, i * D : (i + 1) * D] = row

            constraints.append(LinearConstraint(A, lb, ub))

        elif isinstance(c, opti.LinearInequality):
            # write upper/lowe bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.ones(n_experiments) * c.rhs

            # write constraint as matrix
            lhs = {c.names[i]: c.lhs[i] for i in range(len(c.names))}
            row = np.zeros(D)
            for i, name in enumerate(problem.inputs.names):
                if name in lhs.keys():
                    row[i] = lhs[name]

            A = np.zeros(shape=(n_experiments, D * n_experiments))
            for i in range(n_experiments):
                A[i, i * D : (i + 1) * D] = row

            constraints.append(LinearConstraint(A, lb, ub))

        elif isinstance(c, opti.NonlinearEquality):
            # write upper/lower bound as vector
            lb = np.zeros(n_experiments) - tol
            ub = np.zeros(n_experiments) + tol

            # define constraint evaluation
            def fun(x: np.ndarray) -> float:
                x = pd.DataFrame(
                    x.reshape(len(x) // D, D), columns=problem.inputs.names
                )
                return c(x).to_numpy()

            constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, opti.NonlinearInequality):
            # write upper/lower bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation
            def fun(x: np.ndarray) -> float:
                x = pd.DataFrame(
                    x.reshape(len(x) // D, D), columns=problem.inputs.names
                )
                return c(x).to_numpy()

            constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, opti.NChooseK):
            # write upper bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation
            def fun(x: np.ndarray) -> float:
                x[np.abs(x) < tol] = 0
                x = pd.DataFrame(
                    x.reshape(len(x) // D, D), columns=problem.inputs.names
                )
                return c(x).to_numpy()

            # define constraint gradients
            jac = get_jacobian_NChooseK(c, problem, n_experiments, tol)

            constraints.append(NonlinearConstraint(fun, lb, ub, jac=jac))

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


def get_jacobian_NChooseK(
    constraint: opti.NChooseK,
    problem: opti.Problem,
    n_experiments: int,
    tol: float = 1e-3,
):
    """Returns a function that computes the gradient of a NChooseK constraint.

    Args:
        constraint (opti.NChooseK): NChooseK constraint whose gradient should be computed.
        problem (opti.Problem): problem whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.

    Returns:
        A function that returns the gradient of constraint for a given input.

    """
    D = problem.n_inputs

    def jac(x: np.ndarray) -> np.ndarray:
        """Jacobian for the NChooseK constriant."""
        x = x.reshape(len(x) // D, D)

        # randomly permute columns
        permutation = np.random.permutation(D)
        x = x[:, permutation]

        # find correct order of entries
        ind = np.argsort(-np.abs(x), axis=1)[:, constraint.max_active :]

        # mask for sign
        mask = -2 * np.array(x < 0, dtype=int) + 1

        # set to zero where below threshold
        x[np.abs(x) < tol] = 0

        # set gradient value
        j = np.zeros(shape=x.shape, dtype=int)
        np.put_along_axis(j, ind, 1, axis=1)
        j[x == 0] = 0
        j *= mask

        # invert permutation of columns
        _permutation = np.zeros(D, dtype=int)
        for i in range(D):
            _permutation[permutation[i]] = i
        j = j[:, _permutation]

        # write jacobian into larger matrix (where x is interpreted as a long vector)
        J = np.zeros(shape=(n_experiments, D * n_experiments))
        for i in range(n_experiments):
            J[i, i * D : (i + 1) * D] = j[i]

        return J

    return jac