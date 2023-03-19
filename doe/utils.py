import sys
import warnings
from copy import deepcopy
from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import opti
import pandas as pd
from formulaic import Formula
from opti import Categorical, Discrete
from scipy.optimize import LinearConstraint, NonlinearConstraint

CAT_TOL = 0.1
DISCRETE_TOL = 0.1


class ProblemContext:
    def __init__(self, problem: opti.Problem) -> None:
        """Provider of Context of a Problem. Useful to keep track of relaxed variables, which have their own logic.
        Args:
            problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        """
        self._cat_dict = {}
        self._cat_list = []
        self._discrete_list = []
        self._exclude_list = []
        self._is_relaxed = False
        for input in problem.inputs:
            if isinstance(input, Categorical):
                self._exclude_list.append(input.name)
                self._cat_list.append(input.name)
            if isinstance(input, Discrete):
                self._discrete_list.append(input.name)
        self._problem = deepcopy(problem)
        self._original_problem = problem

    def relax_problem(self) -> None:
        """Transforms the owned opti.problem with Categorical and/or Discrete variables into
        its relaxed version. Categorical variables are transformed into their one-hot encoding
        taking values taking discrete values 0 or 1. Then, one-hot encoded variables are relaxed
        to take values between 0 and 1, while fulfilling the constraint, that they have to
        sum up to 1. Discrete variables are just replaxed with Continuous Variables with
        appropriate bounds.

        Returns:
            None.
        """
        new_inputs = []
        if self._problem.constraints:
            new_constraints = [constraint for constraint in self._problem.constraints]
        else:
            new_constraints = []
        for input in self._problem.inputs:
            if isinstance(input, Categorical):
                cat_names = [
                    col.replace("§", "____")
                    for col in input.to_onehot_encoding(pd.Series("dummy")).columns
                ]
                for name in cat_names:
                    new_inputs.append(opti.Continuous(name, [0, 1]))
                    self._exclude_list.append(name)
                self._cat_dict[input.name] = cat_names
                new_constraints.append(opti.LinearEquality(names=cat_names, rhs=1))
            elif isinstance(input, Discrete):
                new_inputs.append(
                    opti.Continuous(
                        name=input.name, domain=[input.bounds[0], input.bounds[1]]
                    )
                )
            else:
                new_inputs.append(input)
        problem = opti.Problem(
            inputs=new_inputs,
            outputs=self._problem.outputs,
            constraints=new_constraints,
        )
        self._problem = problem
        self._is_relaxed = True
        return None

    def unrelax(self) -> None:
        self._problem = self._original_problem
        self._cat_dict = {}
        self._cat_list = []
        self._discrete_list = []
        self._exclude_list = []
        self._is_relaxed = False
        for input in self._original_problem.inputs:
            if isinstance(input, Categorical):
                self._exclude_list.append(input.name)
                self._cat_list.append(input.name)
            if isinstance(input, Discrete):
                self._discrete_list.append(input.name)

    def transform_onto_original_problem(
        self, feasible_points: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms feasible points of a relaxed problem onto a feasible point
        of the original problem.
        Returns:
            Feasible points of the original problem
        """
        for input in self.original_problem.inputs:
            if isinstance(input, opti.Categorical):
                cat_col = [
                    value2cat(x, input)
                    for _, x in feasible_points[self._cat_dict[input.name]].iterrows()
                ]
                feasible_points[input.name] = cat_col
                feasible_points = feasible_points.drop(
                    self._cat_dict[input.name], axis=1
                )
            if isinstance(input, opti.Discrete):
                discrete_col = [
                    value2discrete(x, input) for x in feasible_points[input.name]
                ]
                feasible_points[input.name] = discrete_col
        return feasible_points

    def transform_onto_relaxed_problem(
        self, feasible_points: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms feasible points of the original problem onto a feasible point
        of the relaxed problem.
        Returns:
            Feasible points of the relaxed problem
        """
        for input in self.original_problem.inputs:
            if isinstance(input, opti.Categorical):
                new_cols = input.to_onehot_encoding(feasible_points[input.name])
                feasible_points = pd.concat(
                    [feasible_points.drop([input.name], axis=1), new_cols], axis=1
                )
        return feasible_points

    @property
    def problem(self) -> opti.Problem:
        return self._problem

    @property
    def is_relaxed(self) -> bool:
        return self._is_relaxed

    @property
    def list_of_categorical_variables(self) -> List[str]:
        return self._cat_list

    @property
    def list_of_variables_without_higher_order_terms(self) -> List[str]:
        return self._exclude_list

    @property
    def has_categoricals(self) -> bool:
        return len(self._cat_list) > 0

    @property
    def has_discrete(self) -> bool:
        return len(self._discrete_list) > 0

    @property
    def has_constraint_with_cats_or_discrete_variables(self) -> bool:
        if self._original_problem.constraints:
            for c in self._original_problem.constraints:
                if c.names is not None:
                    if np.any(
                        [
                            name in self._discrete_list + self._cat_list
                            for name in c.names
                        ]
                    ):
                        return True
        return False

    @property
    def original_problem(self) -> opti.Problem:
        return self._original_problem

    def get_formula_from_string(
        self,
        model_type: Union[str, Formula] = "linear",
        rhs_only: bool = True,
    ) -> Formula:
        return get_formula_from_string(
            model_type=model_type,
            problem_context=self,
            rhs_only=rhs_only,
        )


def value2cat(value: pd.Series, input: opti.Categorical):
    if np.max(value.values) < 1 / len(value.values) + CAT_TOL:
        warnings.warn(
            f"Value too close to decision boundary! Projection of value {np.max(value.values)} to category {input.domain[np.argmax(value.values)]} for categorical {input.name} not within tolerance of {CAT_TOL}."
        )
    return input.domain[np.argmax(value.values)]


def value2discrete(value: np.float64, input: opti.Discrete):
    if abs(input.round(value) - value) > DISCRETE_TOL:
        warnings.warn(
            f"Value too close to decision boundary! Projection of value {value} to discrete value {input.round(value)} for discrete variable {input.name} not within tolerance of {DISCRETE_TOL}."
        )
    return input.round(value)


def get_formula_from_string(
    model_type: Union[str, Formula] = "linear",
    problem_context: Optional[ProblemContext] = None,
    rhs_only: bool = True,
) -> Formula:
    """Reformulates a string describing a model or certain keywords as Formula objects.

    Args:
        model_type (str or Formula): A formula containing all model terms.
        problem_context (ProblemContext): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.
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
            formula = linear_formula(problem_context=problem_context)

        # linear and interactions model
        elif model_type == "linear-and-quadratic":
            formula = linear_and_quadratic_formula(problem_context=problem_context)

        # linear and quadratic model
        elif model_type == "linear-and-interactions":
            formula = linear_and_interactions_formula(problem_context=problem_context)

        # fully quadratic model
        elif model_type == "fully-quadratic":
            formula = fully_quadratic_formula(problem_context=problem_context)

        else:
            formula = model_type + "   "

    formula = Formula(formula[:-3])

    if rhs_only:
        if hasattr(formula, "rhs"):
            formula = formula.rhs

    # set recursion limit to old value
    sys.setrecursionlimit(recursion_limit)

    return formula


def linear_formula(
    problem_context: Optional[ProblemContext],
) -> str:
    """Reformulates a string describing a linear-model or certain keywords as Formula objects.
        formula = model_type + "   "

    Args: problem_context (ProblemContext): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        problem_context is not None
    ), "If the model is described by a keyword a problem must be provided"
    formula = "".join([input.name + " + " for input in problem_context.problem.inputs])
    return formula


def linear_and_quadratic_formula(
    problem_context: Optional[ProblemContext],
) -> str:
    """Reformulates a string describing a linear-and-quadratic model or certain keywords as Formula objects.

    Args: problem_context (ProblemContext): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        problem_context is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.name + " + " for input in problem_context.problem.inputs])
    formula += "".join(
        [
            ""
            if input.name
            in problem_context.list_of_variables_without_higher_order_terms  # exclude h.o. terms for categoricals
            else "{" + input.name + "**2} + "
            for input in problem_context.problem.inputs
        ]
    )
    return formula


def linear_and_interactions_formula(
    problem_context: Optional[ProblemContext],
) -> str:
    """Reformulates a string describing a linear-and-interactions model or certain keywords as Formula objects.

    Args: problem_context (ProblemContext): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        problem_context is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.name + " + " for input in problem_context.problem.inputs])
    for i in range(problem_context.problem.n_inputs):
        for j in range(i):
            # exclude h.o. terms for categoricals
            exlude_flag = (
                problem_context.problem.inputs.names[i]
                in problem_context.list_of_variables_without_higher_order_terms
                and problem_context.problem.inputs.names[j]
                in problem_context.list_of_variables_without_higher_order_terms
            )

            if exlude_flag:
                """"""
            else:
                formula += (
                    problem_context.problem.inputs.names[j]
                    + ":"
                    + problem_context.problem.inputs.names[i]
                    + " + "
                )
    return formula


def fully_quadratic_formula(
    problem_context: Optional[ProblemContext],
) -> str:
    """Reformulates a string describing a fully-quadratic model or certain keywords as Formula objects.

    Args: problem_context (ProblemContext): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        problem_context is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.name + " + " for input in problem_context.problem.inputs])
    for i in range(problem_context.problem.n_inputs):
        for j in range(i):
            # exclude h.o. terms for categoricals
            exlude_flag = (
                problem_context.problem.inputs.names[i]
                in problem_context.list_of_variables_without_higher_order_terms
                and problem_context.problem.inputs.names[j]
                in problem_context.list_of_variables_without_higher_order_terms
            )
            if exlude_flag:
                """"""
            else:
                formula += (
                    problem_context.problem.inputs.names[j]
                    + ":"
                    + problem_context.problem.inputs.names[i]
                    + " + "
                )
    formula += "".join(
        [
            ""
            if input.name
            in problem_context.list_of_variables_without_higher_order_terms  # exclude h.o. terms for categoricals
            else "{" + input.name + "**2} + "
            for input in problem_context.problem.inputs
        ]
    )
    return formula


def n_zero_eigvals(
    problem_context: ProblemContext, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Determine the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""

    # sample points (fulfilling the constraints)
    model_formula = problem_context.get_formula_from_string(
        model_type=model_type, rhs_only=True
    )
    N = len(model_formula.terms) + 3
    X = problem_context.problem.sample_inputs(N)

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

            # define constraint evaluation (and gradient if provided)
            fun = ConstraintWrapper(
                constraint=c, problem=problem, n_experiments=n_experiments, tol=tol
            )

            if c.jacobian_expression is not None:
                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))
            else:
                constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, opti.NonlinearInequality):
            # write upper/lower bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation (and gradient if provided)
            fun = ConstraintWrapper(
                constraint=c, problem=problem, n_experiments=n_experiments, tol=tol
            )

            if c.jacobian_expression is not None:
                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))
            else:
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
        n_experiments: int = 0,
        tol: float = 1e-3,
    ) -> None:
        """
        Args:
            constraint (opti.constraint.Constraint): opti constraint to be called
            problem (opti.Problem): problem the constraint belongs to
            n_experiments (int): number of experiments
            tol (float): tolerance for constraint violation. Default value is 1e-3.
        """
        self.constraint = constraint
        self.tol = tol
        self.names = problem.inputs.names
        self.D = problem.n_inputs
        self.n_experiments = n_experiments
        if constraint.names is None:
            raise ValueError(
                f"The features attribute of constraint {constraint} is not set, but has to be set."
            )
        self.constraint_name_indices = np.searchsorted(
            self.names, self.constraint.names
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """call constraint with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)
        violation = self.constraint(x).to_numpy()
        violation[np.abs(violation) < self.tol] = 0
        return violation

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """call constraint gradient with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)
        jacobian_compressed = self.constraint.jacobian(x).to_numpy()

        jacobian = np.zeros(shape=(self.n_experiments, self.D * self.n_experiments))
        rows = np.repeat(
            np.arange(self.n_experiments), len(self.constraint_name_indices)
        )
        cols = np.repeat(
            self.D * np.arange(self.n_experiments), len(self.constraint_name_indices)
        ).reshape((self.n_experiments, len(self.constraint_name_indices)))
        cols = (cols + self.constraint_name_indices).flatten()

        jacobian[rows, cols] = jacobian_compressed.flatten()

        return jacobian


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
    X: np.ndarray,
    problem: opti.Problem,
    delta: float = 1e-9,
    n_samples: int = 1e4,
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

    # X has to contain numerical values for metrics
    # thus transform to one-hot-encoded matrix
    # with properly transformed problem
    problem_context = ProblemContext(problem=problem)
    if problem_context.has_categoricals:
        problem_context.relax_problem()
        X = problem_context.transform_onto_relaxed_problem(X)

    # try to determine G-efficiency
    try:
        g_eff = g_efficiency(
            X,
            problem_context.problem,
            delta,
            n_samples,
        )

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
    for c in nchoosek_constraints:
        for name in np.unique(c.names):
            if problem.inputs[name].domain[0] > 0 or problem.inputs[name].domain[1] < 0:
                raise ValueError(
                    f"Constraint {c} cannot be formulated as bounds. 0 must be inside the \
                    domain of the affected decision variables."
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
