import sys
import warnings
from itertools import combinations
from typing import List, Optional, Union

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
    # set maximum recursion depth to higher value
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(2000)

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

    # set recursion limit to old value
    sys.setrecursionlimit(recursion_limit)

    return formula


def n_zero_eigvals(
    problem: opti.Problem, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Determine the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""
    # sample points (fulfilling the constraints)
    model_formula = get_formula_from_string(
        model_type=model_type, problem=problem, rhs_only=True
    )
    N = len(model_formula.terms) + 3
    X = problem.sample_inputs(N)

    # compute eigenvalues of information matrix
    A = model_formula.get_model_matrix(X)
    eigvals = np.abs(np.linalg.eigvalsh(A.T @ A))

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def constraints_as_scipy_constraints(
    problem: opti.Problem,
    n_experiments: int,
    tol: float = 1e-3,
    linearize_NChooseK: bool = False,
) -> List:
    """Formulates opti constraints as scipy constraints.

    Args:
        problem (opti.Problem): problem whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.
        linearize_NChooseK (bool): Tries to replace NChooseK constraints by linear constraints if set
            to True. For details see nchoosek_constraint_as_scipy_linear_constraint() function.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.
    """
    D = problem.n_inputs

    # check if nchoosek constraint linearization can be done
    if linearize_NChooseK:
        if problem.n_constraints > 0:
            if any([isinstance(c, opti.NChooseK) for c in problem.constraints]):
                check_nchoosek_constraints_linearizable(problem)
                warnings.warn(
                    "linearized versions of NChooseK constraints are based on heuristics \
                    and may result in unexpected behavior."
                )

    # reformulate constraints
    constraints = []
    if problem.constraints is None:
        return constraints
    for c in problem.constraints:
        if isinstance(c, opti.LinearEquality):
            # write lower/upper bound as vector
            lb = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.lhs) - tol)
            ub = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.lhs) + tol)

            # write constraint as matrix
            lhs = {
                c.names[i]: c.lhs[i] / np.linalg.norm(c.lhs)
                for i in range(len(c.names))
            }
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
            ub = np.ones(n_experiments) * c.rhs / np.linalg.norm(c.lhs)

            # write constraint as matrix
            lhs = {
                c.names[i]: c.lhs[i] / np.linalg.norm(c.lhs)
                for i in range(len(c.names))
            }
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
            fun = ConstraintWrapper(constraint=c, problem=problem, tol=tol)

            constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, opti.NonlinearInequality):
            # write upper/lower bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation
            fun = ConstraintWrapper(constraint=c, problem=problem, tol=tol)

            constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, opti.NChooseK):

            if linearize_NChooseK:
                # convert NChooseK constraint to linear constraint
                constr = nchoosek_constraint_as_scipy_linear_constraint(
                    constraint=c,
                    names=problem.inputs.names,
                    n_experiments=n_experiments,
                    tol=tol,
                )

                constraints.append(constr)

            else:
                # write upper/lower bound as vector
                lb = -np.inf * np.ones(n_experiments)
                ub = np.zeros(n_experiments)

                # define constraint evaluation
                fun = ConstraintWrapper(constraint=c, problem=problem, tol=tol)

                # define constraint gradients
                jac = JacobianNChooseK(
                    constraint=c, problem=problem, n_experiments=n_experiments, tol=tol
                )

                constraints.append(NonlinearConstraint(fun, lb, ub, jac=jac))

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


class ConstraintWrapper:
    """Wrapper for opti constraint calls using flattened numpy arrays instead of ."""

    def __init__(
        self,
        constraint: opti.constraint.Constraint,
        problem: opti.Problem,
        tol: float = 1e-3,
    ) -> None:
        """
        Args:
            constraint (opti.constraint.Constraint): opti constraint to be called
            problem (opti.Problem): problem the constraint belongs to
            tol (float): tolerance for constraint violation. Default value is 1e-3.
        """
        self.constraint = constraint
        self.tol = tol
        self.names = problem.inputs.names
        self.D = problem.n_inputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "call constraint with flattened numpy array"
        x[np.abs(x) < self.tol] = 0
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)
        return self.constraint(x).to_numpy()


class JacobianNChooseK:
    """Jacobian for NChooseK constraints."""

    def __init__(
        self,
        constraint: opti.NChooseK,
        problem: opti.Problem,
        n_experiments: int,
        tol: float = 1e-3,
    ) -> None:
        """
        Args:
            constraint (opti.NChooseK): NChooseK constraint whose gradient should be computed.
            problem (opti.Problem): problem whose constraints should be formulated as scipy constraints.
            n_experiments (int): Number of instances of inputs for problem that are evaluated together.
            tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.

        Returns:
            A function that returns the gradient of constraint for a given input.

        """
        self.constraint = constraint
        self.n_experiments = n_experiments
        self.tol = tol
        self.D = problem.n_inputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Jacobian for the NChooseK constriant."""
        x = x.reshape(len(x) // self.D, self.D)

        # randomly permute columns
        permutation = np.random.permutation(self.D)
        x = x[:, permutation]

        # find correct order of entries
        ind = np.argsort(-np.abs(x), axis=1)[:, self.constraint.max_active :]

        # mask for sign
        mask = -2 * np.array(x < 0, dtype=int) + 1

        # set to zero where below threshold
        x[np.abs(x) < self.tol] = 0

        # set gradient value
        j = np.zeros(shape=x.shape, dtype=int)
        np.put_along_axis(j, ind, 1, axis=1)
        j[x == 0] = 0
        j *= mask

        # invert permutation of columns
        _permutation = np.zeros(self.D, dtype=int)
        for i in range(self.D):
            _permutation[permutation[i]] = i
        j = j[:, _permutation]

        # write jacobian into larger matrix (where x is interpreted as a long vector)
        J = np.zeros(shape=(self.n_experiments, self.D * self.n_experiments))
        for i in range(self.n_experiments):
            J[i, i * self.D : (i + 1) * self.D] = j[i]

        return J


def d_optimality(X: np.ndarray, tol=1e-9) -> float:
    """Compute ln(1/|X^T X|) for a model matrix X (smaller is better).
    The covariance of the estimated model parameters for $y = X beta + epsilon $is
    given by $Var(beta) ~ (X^T X)^{-1}$.
    The determinant |Var| quantifies the volume of the confidence ellipsoid which is to
    be minimized.
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > tol]
    return np.sum(np.log(eigenvalues))


def a_optimality(X: np.ndarray, tol=1e-9) -> float:
    """Compute the A-optimality for a model matrix X (smaller is better).
    A-optimality is the sum of variances of the estimated model parameters, which is
    the trace of the covariance matrix $X.T @ X^-1$.

    F is symmetric positive definite, hence the trace of (X.T @ X)^-1 is equal to the
    the sum of inverse eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > tol]
    return np.sum(1 / eigenvalues)


def g_efficiency(
    X: np.ndarray, problem: opti.Problem, delta: float = 1e-9, n_samples: int = 1e4
) -> float:
    """Compute the G-efficiency for a model matrix X.
    G-efficiency is proportional to p/(n*d) where p is the number of model terms,
    n is the number of runs and d is the maximum relative prediction variance over
    the set of runs.
    """

    # number of runs and model terms
    n, p = X.shape

    # take large sample from the design space
    Y = problem.sample_inputs(int(n_samples)).to_numpy()

    # variance over set of runs
    D = Y @ np.linalg.inv(X.T @ X + delta * np.eye(p)) @ Y.T
    d = np.max(np.diag(D))

    G_eff = 100 * p / (n * d)
    return G_eff


def metrics(
    X: np.ndarray,
    problem: opti.Problem,
    tol: float = 1e-9,
    delta: float = 1e-9,
    n_samples: int = 1e4,
) -> pd.Series:
    """Returns a series containing D-optimality, A-optimality and G-efficiency
    for a model matrix X

    Args:
        X (np.ndarray): model matrix for which the metrics are determined
        problem (opti.Problem): problem definition containing the constraints of the design space.
        tol (float): cutoff value for eigenvalues of the information matrix in
            D- and A- optimality computation. Default value is 1e-9.
        delta (float): regularization parameter in G-efficiency computation.
            Default value is 1e-9
        n_samples (int): number of samples used to determine G-efficiency. Default value is 1e4.

    Returns:
        A pd.Series containing the values for the three metrics.
    """

    # try to determine G-efficiency
    try:
        g_eff = g_efficiency(X, problem, delta, n_samples)

    except Exception:
        warnings.warn(
            "Sampling of points fulfilling this problem's constraints is not implemented. \
            G-efficiency can't be determined."
        )
        g_eff = 0

    return pd.Series(
        {
            "D-optimality": d_optimality(X, tol),
            "A-optimality": a_optimality(X, tol),
            "G-efficiency": g_eff,
        }
    )


def nchoosek_constraint_as_scipy_linear_constraint(
    constraint: opti.NChooseK, names: List, n_experiments: int, tol: float = 1e-3
) -> LinearConstraint:
    """Determines a linear constraint that is closely related to the NChooseK constraint.
    Only works under strict restrictions: The parameters in the constraints' names attribute
    must have a domain of the form [0, xu] where xu > 0. If more than one NChooseK constraint
    of a problem refers to the same parameter, this method cannot be used. For each experiment it
    chooses (N-K) inputs from the NChooseK constraints' names list and creates a linear constraint
    that forces these inputs to be zero.

    Args:
        constraint (opti.NChooseK): constraint to be formulated as linear constraint.
        names (List): list of names of the input values of the corresponding problem
        n_experiments (int): number of experiments for the design to be determined.


    Returns:
        A scipy.LinearConstraint being a sufficient condition for the NChooseK constraint if
        all conditions on the NChooseK constraint are met.
    """
    n_inactive = len(constraint.names) - constraint.max_active

    # find indices of constraint.names in names
    ind = [i for i, p in enumerate(names) if p in constraint.names]

    # find and shuffle all combinations of elements of ind of length max_active
    ind = np.array([c for c in combinations(ind, r=n_inactive)])
    np.random.shuffle(ind)

    # set up linear constraint matrix
    A = np.zeros(shape=(1, len(names) * n_experiments))

    for i in range(n_experiments):
        ind_vanish = ind[i % len(ind)]
        A[0, ind_vanish + i * len(names)] = 1
    A /= np.sqrt(n_inactive * n_experiments)

    # write lower/upper bounds as vector
    lb = -np.inf * np.ones(1)
    ub = np.zeros(1) + tol

    return LinearConstraint(A, lb, ub)


def check_nchoosek_constraints_linearizable(problem: opti.Problem) -> None:
    # collect NChooseK constraints
    if problem.n_constraints == 0:
        return

    nchoosek_constraints = []
    for c in problem.constraints:
        if isinstance(c, opti.NChooseK):
            nchoosek_constraints.append(c)

    if len(nchoosek_constraints) == 0:
        return

    # check if the domains of all NCHooseK constraints are compatible to linearization
    parameter_names = np.unique(np.concatenate([c.names for c in nchoosek_constraints]))
    for name in parameter_names:
        if problem.inputs[name].domain[0] != 0:
            raise ValueError(
                f"Constraint {c} cannot be linearized. Lower bound of domain must be 0."
            )

    # check if the parameter names of two nchoose overlap
    for c in nchoosek_constraints:
        for _c in nchoosek_constraints:
            if c != _c:
                for name in c.names:
                    if name in _c.names:
                        raise ValueError(
                            f"Problem {problem} cannot be used for linearization. \
                            names attribute of NChooseK constraints must be pairwise disjoint."
                        )
