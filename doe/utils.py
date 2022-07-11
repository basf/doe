from ast import Str
import sys
import warnings
from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import opti
from opti import Continuous,Categorical
import pandas as pd
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint


class ProblemHelper():
    def __init__(self, problem: opti.Problem)-> None:
        """ Helper for Problem transformations
        Args:
            problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        """
        self._problem = problem

    def transform2relaxed(self)-> opti.Problem:
        """Transforms an opti.problem with Categorical variables into its relaxed version.
            Categorical variables are transformed into their one-hot encoding taking values taking
            discrete values 0 or 1. Then, one-hot encoded variables are relaxed to take values
            between 0 and 1, while fulfilling the constraint, that they have to sum up to 1.

        Returns:
            A string describing the model that was given as string or keyword.
        """
        new_inputs = []
        if self._problem.constraints:
            new_constraints = self._problem.constraints
        else:
            new_constraints = []
        for input in self._problem.inputs:
            if isinstance(input,Continuous):
                new_inputs.append(input)
            if isinstance(input, Categorical):
                var_names = [col.replace('§','') for col in input.to_onehot_encoding(pd.Series("dummy")).columns]
                for name in var_names:
                    new_inputs.append(opti.Continuous(name,[0,1]))
                new_constraints.append(opti.LinearEquality(names=var_names, rhs=1))
        problem = opti.Problem(
            inputs=new_inputs,
            outputs=self._problem.outputs,
            constraints=new_constraints
        )
        return problem

class FormulaProvider():
    def __init__(
        self,
        model_type: Union[str, Formula] = "linear",
        problem: Optional[opti.Problem] = None,
        rhs_only: bool = True,
    ) -> None:
        """ Provider of formulas
        Args:
            model_type (str or Formula): A formula containing all model terms.
            problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
            Only needed if the model is defined by a keyword
            rhs_only (bool): The function returns only the right hand side of the formula if set to True.

        """
        self.model_type = model_type
        self.problem = problem
        self.rhs_only = rhs_only

    def get_formula_from_string(
        self,
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

        if isinstance(self.model_type, Formula):
            return self.model_type

        # build model if a keyword and a problem are given.
        else:
            # linear model
            if self.model_type == "linear":
                formula = self.linear_formula()

            # linear and interactions model
            elif self.model_type == "linear-and-quadratic":
                formula = self.linear_and_quadratic_formula()

            # linear and quadratic model
            elif self.model_type == "linear-and-interactions":
                formula = self.linear_and_interactions_formula()

            # fully quadratic model
            elif self.model_type == "fully-quadratic":
                formula = self.fully_quadratic_formula()

            else:
                formula = self.model_type + "   "
        print(formula)
        formula = Formula(formula[:-3])

        if self.rhs_only:
            if hasattr(formula, "rhs"):
                formula = formula.rhs

        # set recursion limit to old value
        sys.setrecursionlimit(recursion_limit)

        return formula


    def linear_formula(self,) ->str :
        """Reformulates a string describing a linear-model or certain keywords as Formula objects.

        Args:
            None

        Returns:
            A string describing the model that was given as string or keyword.
        """
        assert (
                self.problem is not None
            ), "If the model is described by a keyword a problem must be provided"
        formula =  "".join([input.name + " + " for input in self.problem.inputs])
        return formula


    def linear_and_quadratic_formula(self,) ->str :
        """Reformulates a string describing a linear-and-quadratic model or certain keywords as Formula objects.

        Args:
            None

        Returns:
            A string describing the model that was given as string or keyword.
        """
        assert (
                    self.problem is not None
                ), "If the model is described by a keyword a problem must be provided."
        formula = "".join([input.name + " + " for input in self.problem.inputs])
        formula += "".join(
            ["{" + input.name + "**2} + " for input in self.problem.inputs]
        )
        return formula

    def linear_and_interactions_formula(self,) ->str:
        """Reformulates a string describing a linear-and-interactions model or certain keywords as Formula objects.

        Args:
            None

        Returns:
            A string describing the model that was given as string or keyword.
        """
        assert (
            self.problem is not None
        ), "If the model is described by a keyword a problem must be provided."
        formula = "".join([input.name + " + " for input in self.problem.inputs])
        for i in range(self.problem.n_inputs):
            for j in range(i):
                formula += (
                    self.problem.inputs.names[j] + ":" + self.problem.inputs.names[i] + " + "
                )
        return formula

    def fully_quadratic_formula(self,)-> str:
        """Reformulates a string describing a fully-quadratic model or certain keywords as Formula objects.

        Args:
            None

        Returns:
            A string describing the model that was given as string or keyword.
        """
        assert (
            self.problem is not None
        ), "If the model is described by a keyword a problem must be provided."
        formula = "".join([input.name + " + " for input in self.problem.inputs])
        for i in range(self.problem.n_inputs):
            for j in range(i):
                formula += (
                    self.problem.inputs.names[j] + ":" + self.problem.inputs.names[i] + " + "
                )
        formula += "".join(
            ["{" + input.name + "**2} + " for input in self.problem.inputs]
        )
        return formula

def n_zero_eigvals(
    problem: opti.Problem, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Determine the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""

    formula_provider =  FormulaProvider(model_type=model_type, problem=problem, rhs_only=True)
    # sample points (fulfilling the constraints)
    model_formula = formula_provider.get_formula_from_string(
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
) -> List:
    """Formulates opti constraints as scipy constraints. Ignores NchooseK constraints
    (these can be formulated as bounds).

    Args:
        problem (opti.Problem): problem whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.
    """
    D = problem.n_inputs

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
            pass

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


def check_nchoosek_constraints_as_bounds(problem: opti.Problem) -> None:
    """Checks if NChooseK constraints of problem can be formulated as bounds.

    Args:
        problem (opti.Problem): problem whose NChooseK constraints should be checked
    """
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
        if problem.inputs[name].domain[0] > 0 or problem.inputs[name].domain[1] < 0:
            raise ValueError(
                f"Constraint {c} cannot be formulated as bounds. 0 must be inside the domain of the affected decision variables."
            )

    # check if the parameter names of two nchoose overlap
    for c in nchoosek_constraints:
        for _c in nchoosek_constraints:
            if c != _c:
                for name in c.names:
                    if name in _c.names:
                        raise ValueError(
                            f"Problem {problem} cannot be used for formulation as bounds. \
                            names attribute of NChooseK constraints must be pairwise disjoint."
                        )


def nchoosek_constraints_as_bounds(
    problem: opti.Problem,
    n_experiments: int,
) -> List:
    """Determines the box bounds for the decision variables

    Args:
        problem (opti.Problem): problem to find the bounds for.
        n_experiments (int): number of experiments for the design to be determined.

    Returns:
        A list of tuples containing bounds that respect NChooseK constraint imposed
        onto the decision variables.
    """
    check_nchoosek_constraints_as_bounds(problem)

    # bounds without NChooseK constraints
    bounds = np.array([(p.bounds) for p in problem.inputs] * n_experiments)

    if problem.n_constraints > 0:
        for constraint in problem.constraints:
            if isinstance(constraint, opti.NChooseK):

                n_inactive = len(constraint.names) - constraint.max_active

                # find indices of constraint.names in names
                ind = [
                    i
                    for i, p in enumerate(problem.inputs.names)
                    if p in constraint.names
                ]

                # find and shuffle all combinations of elements of ind of length max_active
                ind = np.array([c for c in combinations(ind, r=n_inactive)])
                np.random.shuffle(ind)

                # set bounds to zero in each experiments for the variables that should be inactive
                for i in range(n_experiments):
                    ind_vanish = ind[i % len(ind)]
                    bounds[ind_vanish + i * len(problem.inputs.names), :] = [0, 0]

    # convert bounds to list of tuples
    bounds = [(b[0], b[1]) for b in bounds]

    return bounds
