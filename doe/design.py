from typing import Callable, Optional, Union

import numpy as np
import opti
import pandas as pd
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint, basinhopping


# TODO: Umschreiben, sodass problem ein optionales Argument ist
def get_formula_from_string(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    rhs_only=True,
) -> Formula:
    """Reformulates a string describing a model or certain keywords as Formula objects.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        rhs_only (bool): The function returns only the right hand side of the formula if set to True.

    Returns:
        A Formula object describing the model that was given as string or keyword.
    """
    if isinstance(model_type, Formula):
        return model_type

    else:
        if model_type == "linear":
            formula = "".join([input.name + " + " for input in problem.inputs])

        elif model_type == "linear-and-quadratic":
            formula = "".join([input.name + " + " for input in problem.inputs])
            formula += "".join(
                ["{" + input.name + "**2} + " for input in problem.inputs]
            )

        elif model_type == "linear-and-interactions":
            formula = "".join([input.name + " + " for input in problem.inputs])
            for i in range(problem.n_inputs):
                for j in range(i):
                    formula += (
                        problem.inputs.names[j] + ":" + problem.inputs.names[i] + " + "
                    )

        elif model_type == "fully-quadratic":
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


def n_ignore_eigvals(
    problem: opti.Problem, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Computes the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""
    model_formula = get_formula_from_string(problem, model_type)
    N = len(model_formula.terms) + 3

    model_matrix = model_formula.get_model_matrix(problem.sample_inputs(N))
    eigvals = np.abs(np.linalg.eigvals(model_matrix.T @ model_matrix))

    return len(eigvals) - len(eigvals[eigvals > epsilon])


# David fragen/selber denken: Du wolltest explizit, dass slogdet verwendet wird,
# das geht jetzt nicht, da einige EW'en ausgelassen werden sollen --> in Ordnung? Gibt es was besseres?
def logD(A: np.ndarray, n_ignore_eigvals: int) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    eigvals = np.sort(np.linalg.eigvals(A.T @ A))[n_ignore_eigvals:]
    return np.sum(np.log(eigvals))


def get_callback(
    problem: opti.Problem,
    n_max_min_found: int = 100,
    n_max_no_change: int = 10,
    tol: float = 1e-3,
    verbose: bool = False,
) -> Callable:
    """Returns a callback function for basinhopping from scipy.optimize.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        n_max_min_found (int): Number of found minima after which the algorithm will stop. Default value is 100.
        n_max_no_change (int): Algorithm stops if the best minimum has not improved for the last n_max_no_change
            minima. Default value is 10.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.
        verbose: prints status messages if set to True. Default value is False.

    Returns:
        A callback function implementing status messages and the stop criteria for n_max_no_change, n_max_min_found.
    """

    # define constraint violation function
    if problem.constraints is not None:

        def constraint_violation(x):

            D = problem.n_inputs
            A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)

            penalty = 0
            for constraint in problem.constraints:
                if any(
                    [
                        isinstance(constraint, c)
                        for c in [
                            opti.LinearEquality,
                            opti.NonlinearEquality,
                            opti.NChooseK,
                        ]
                    ]
                ):
                    eq_penalty = np.abs((constraint(A)).values) - tol
                    eq_penalty[eq_penalty <= 0] = 0
                    penalty += np.sum(eq_penalty)
                else:
                    penalty += np.sum(constraint(A).clip(0, None).values)
            return penalty

    else:

        def constraint_violation(x):
            return 0

    def callback(x, f, accept):
        callback.n_calls += 1

        if f < callback.f_opt:
            # reset n_no_change counter
            callback.n_no_change = 1

            # update objective value
            callback.f_opt = f

            # compute constraint violation
            callback.cv_opt = constraint_violation(x)
        else:
            # increase n_no_change counter
            callback.n_no_change += 1

        if verbose:
            # print status if verbose
            if callback.n_calls == 1:
                print(
                    "--------------------------------------------------------------------"
                )
                print(
                    "| n_optima_found |        logD_opt        |           CV           |"
                )
                print(
                    "--------------------------------------------------------------------"
                )

            n_calls = str(callback.n_calls)
            while len(n_calls) < 16:
                n_calls += " "

            logD_opt = str(-callback.f_opt)
            while len(logD_opt) < 24:
                logD_opt += " "

            cv_opt = str(callback.cv_opt)
            while len(cv_opt) < 24:
                cv_opt += " "

            print("|" + n_calls + "|" + logD_opt + "|" + cv_opt + "|")

        # check stop criteria
        if callback.n_calls >= callback.n_max_min_found:
            callback.stop = True

        if callback.n_no_change >= callback.n_max_no_change:
            callback.stop = True

        return callback.stop

    callback.n_max_min_found = n_max_min_found
    callback.n_max_no_change = n_max_no_change
    callback.n_calls = 0
    callback.f_opt = np.inf
    callback.cv_opt = np.inf
    callback.verbose = verbose
    callback.n_no_change = 0
    callback.stop = False

    return callback


def get_objective(
    problem: opti.Problem,
    model_type: Union[str, Formula],
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
    num_ignore_eigvals = n_ignore_eigvals(problem, model_type)

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD(A.to_numpy(), num_ignore_eigvals)
        return obj

    return objective


def constraints_as_scipy_constraints(
    problem: opti.Problem, n_experiments: int, tol: float = 1e-3
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
            ub = np.zeros(n_experiments) + tol

            # define constraint evaluation
            def fun(x: np.ndarray) -> float:
                x = pd.DataFrame(
                    x.reshape(len(x) // D, D), columns=problem.inputs.names
                )
                return c(x).to_numpy()

            constraints.append(NonlinearConstraint(fun, lb, ub))

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


def optimal_design(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    n_max_min_found: int = 1e2,
    n_max_no_change: int = 10,
    tol: float = 1e-3,
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
        verbose (bool): status messages are shown regularily if set to True.

    Returns:
        A pd.DataFrame object containing the best found input parameters for the experiments.
    """
    # TODO: unterstützung für categorical inputs
    D = problem.n_inputs
    model_formula = get_formula_from_string(problem, model_type, rhs_only=True)

    # David fragen/selbst nachdenken: so in Ordnung?
    if n_experiments is None:
        n_experiments = (
            len(model_formula.terms) - n_ignore_eigvals(problem, model_formula) + 3
        )

    # get callback for stop criterion an status messages
    callback = get_callback(
        problem=problem,
        n_max_min_found=n_max_min_found,
        n_max_no_change=n_max_no_change,
        tol=tol,
        verbose=verbose,
    )

    # get objective function
    objective = get_objective(problem, model_type)

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(problem, n_experiments, tol)

    # write accept_test function to test if a step is inside the bounds
    def accept_test(**kwargs):
        x = kwargs["x_new"]
        test_max = bool(np.all(x <= accept_test.ub))
        test_min = bool(np.all(x >= accept_test.lb))
        return test_max and test_min

    bounds = np.array([(p.bounds) for p in problem.inputs] * n_experiments).T
    accept_test.lb = bounds[0]
    accept_test.ub = bounds[1]

    # do the optimization
    result = basinhopping(
        objective,
        x0=problem.sample_inputs(n_experiments).values.reshape(-1),
        minimizer_kwargs={
            "method": "SLSQP",
            "bounds": [(p.bounds) for p in problem.inputs] * n_experiments,
            "constraints": constraints,
        },
        callback=callback,
        accept_test=accept_test,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A


# TODO:
# Sampling mit NChooseK Constraint
# bessere Stoppkriterien?

# from itertools import combinations

# def num_experiments(
#     problem: opti.Problem, model_type: str = "linear", n_dof: int = 3
# ) -> int:
#     """Determine the number of experiments needed to fit a model to a given problem.

#     Args:
#         problem (opti.Problem): Specification of the design and objective space.
#         model_type (str, optional): Type of model. Defaults to "linear".
#         n_dof (int, optional): Additional experiments to add degrees of freedom to a resulting model fit. Defaults to 3.

#     Returns:
#         int: Number of experiments.
#     """
#     continuous = [p for p in problem.inputs if isinstance(p, opti.Continuous)]
#     discretes = [p for p in problem.inputs if isinstance(p, opti.Discrete)]
#     categoricals = [p for p in problem.inputs if isinstance(p, opti.Categorical)]

#     constraints = problem.constraints
#     if constraints is None:
#         constraints = []
#     equalities = [c for c in constraints if isinstance(c, opti.LinearEquality)]

#     # number of linearly independent numeric variables
#     d = len(continuous) + len(discretes) - len(equalities) # --> Stimmt das nicht eigentlich nur, wenn alle Gleichungen linear unabhängig sind?

#     # categoricals are encoded using dummy variables, hence a categorical with 4 levels requires 3 experiments in a linear model.
#     dummy_levels = [len(p.domain) - 1 for p in categoricals]

#     n_linear = 1 + d + sum(dummy_levels)
#     n_quadratic = d  # dummy variables are not squared
#     n_interaction = d * (d - 1) / 2
#     n_interaction += d * sum(dummy_levels)
#     n_interaction += sum([np.prod(k) for k in combinations(dummy_levels, 2)])

#     if model_type == "linear":
#         n_experiments = n_linear + n_dof
#     elif model_type == "linear-and-quadratic":
#         n_experiments = n_linear + n_quadratic + n_dof
#     elif model_type == "linear-and-interactions":
#         n_experiments = n_linear + n_interaction + n_dof
#     elif model_type == "fully-quadratic":
#         n_experiments = n_linear + n_interaction + n_quadratic + n_dof
#     else:
#         raise Exception(f"Unknown model type {model_type}")
#     return int(n_experiments)
