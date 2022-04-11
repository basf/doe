from typing import Callable, Optional, Union

import numpy as np
import opti
import pandas as pd
from formulaic import Formula
from scipy.optimize import basinhopping


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
def logD_(A: np.ndarray, n_ignore_eigvals: int) -> float:
    """Computes the sum of the log of A.T @ A ignoring the smallest num_ignore_eigvals eigenvalues."""
    eigvals = np.sort(np.linalg.eigvals(A.T @ A))[n_ignore_eigvals:]
    return np.sum(np.log(eigvals))


# TODO: ggf. safety margins für inequalities entfernen
def get_constraint_violation(problem: opti.Problem, tol=1e-3) -> Callable:
    """Returns a function that evaluates the constraint violation of an opti Problem with safety margins."""
    if problem.constraints is not None:

        def cv(A):
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
                    penalty += np.sum((constraint(A) - tol).clip(0, None).values)
            return penalty

    else:

        def cv(A):
            return 0

    return cv


# TODO: einbauen: ablehnen, falls ausserhalb der constraints?
def get_callback(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_max_min_found: int = 100,
    n_max_no_change: int = 25,
    tol: float = 1e-3,
    verbose: bool = False,
) -> Callable:
    """Returns a callback function for basinhopping from scipy.optimize.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        n_max_min_found (int): Number of found minima after which the algorithm will stop. Default value is 100.
        n_max_no_change (int): Algorithm stops if the best minimum has not improved for the last n_max_no_change
            minima. Default value is 25.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.
        verbose: prints status messages if set to True. Default value is False.

    Returns:
        A callback function implementing status messages and the stop criteria for n_max_no_change, n_max_min_found.
    """

    def callback(x, f, accept):
        callback.n_calls += 1
        model_formula = get_formula_from_string(problem, model_type)
        constraint_violation = get_constraint_violation(problem, tol=tol)
        num_ignore_eigvals = n_ignore_eigvals(problem, model_formula)

        if f < callback.f_opt:
            # reset n_no_change counter
            callback.n_no_change = 1

            # update objective value
            callback.f_opt = f

            # compute value for D criterion
            D = problem.n_inputs
            A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
            A = model_formula.get_model_matrix(A)
            callback.logD_opt = logD_(A.to_numpy(), num_ignore_eigvals)

            # compute constraint violation
            callback.cv_opt = constraint_violation(A)
        else:
            # increase n_no_change counter
            callback.n_no_change += 1

        if verbose:
            # print status if verbose
            if callback.n_calls == 1:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print(
                    "| n_optima_found |        logD_opt        |           CV           |         f_opt         |"
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

            n_calls = str(callback.n_calls)
            while len(n_calls) < 16:
                n_calls += " "

            logD_opt = str(callback.logD_opt)
            while len(logD_opt) < 24:
                logD_opt += " "

            cv_opt = str(callback.cv_opt)
            while len(cv_opt) < 24:
                cv_opt += " "

            f_opt = str(callback.f_opt)
            while len(f_opt) < 23:
                f_opt += " "

            print("|" + n_calls + "|" + logD_opt + "|" + cv_opt + "|" + f_opt + "|")

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
    callback.logD_opt = np.inf
    callback.cv_opt = np.inf
    callback.verbose = verbose
    callback.n_no_change = 0
    callback.stop = False

    return callback


def get_objective(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    tol: float = 1e-3,
    constraint_weight: float = 40,
) -> Callable:
    """Returns a function that computes the combined objective.

    Args:
        problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        tol (float): tolarance for the computation of constraint violation. Default value is 1e-3.
        constraint_weight (float): relative weight factor between constraint violation and objective.
            Default value is 40.

    Returns:
        A function computing the weighted sum of the logD value and the constraint violation
        for a given input.

    """
    D = problem.n_inputs
    model_formula = get_formula_from_string(problem, model_type, rhs_only=True)
    num_ignore_eigvals = n_ignore_eigvals(problem, model_type)
    constraint_violation = get_constraint_violation(problem, tol=tol)

    # define objective function
    def objective(x):
        # evaluate model terms
        A = pd.DataFrame(x.reshape(len(x) // D, D), columns=problem.inputs.names)
        A = model_formula.get_model_matrix(A)

        # compute objective value
        obj = -logD_(A.to_numpy(), num_ignore_eigvals)
        obj += constraint_weight * constraint_violation(A)
        return obj

    return objective


# TODO: testen
def optimal_design(
    problem: opti.Problem,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    n_max_min_found: int = 1e2,
    n_max_no_change: int = 10,
    tol: float = 1e-3,
    ctol: float = 1e-4,
    constraint_weighting: Optional[float] = None,
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
            not improved the best discovered minimum. Default value is 25.
        tol (float): tolarance for the computation of constraint violation. Default value is 1e-3.
        ctol (float): tolerance for constraint violation when searching for a good constraint weighting.
            Only meaningful if no value for constraint_weighting is given.
        constraint_weighting (float): optional float parameter. If a value is given, automatic
            determination of the constraint weighting will not be done. Instead, the given value will be used.
        verbose (bool): status messages are shown regularily if set to True.

    Returns:
        A pd.DataFrame object containing the best found input parameters for the experiments.
    """
    # TODO: unterstützung für categorical inputs
    # TODO: optimale anzahl an Experimenten N finden
    D = problem.n_inputs
    model_formula = get_formula_from_string(problem, model_type, rhs_only=True)

    # David fragen/selbst nachdenken: so in Ordnung?
    if n_experiments is None:
        n_experiments = (
            len(model_formula.terms) - n_ignore_eigvals(problem, model_formula) + 3
        )

    # find correct constraint weighting
    if constraint_weighting is None:

        if verbose:
            print("Searching for a suitable constraint weighting.")

        constraint_weighting = 0.5
        constraint_violation = np.inf
        compute_constraint_violation = get_constraint_violation(problem, tol=tol)

        while constraint_violation > ctol:

            constraint_weighting *= 2
            if verbose:
                print(f"Trying constraint weighting: {constraint_weighting}.")

            callback = get_callback(
                problem, model_formula, n_max_min_found=2, tol=tol, verbose=verbose
            )
            objective = get_objective(problem, model_type, tol, constraint_weighting)
            result = basinhopping(
                objective,
                x0=problem.sample_inputs(n_experiments).values.reshape(-1),
                minimizer_kwargs={
                    "method": "SLSQP",
                    "bounds": [(p.bounds) for p in problem.inputs] * n_experiments,
                },
                callback=callback,
            )
            A = pd.DataFrame(
                result["x"].reshape(n_experiments, D), columns=problem.inputs.names
            )
            constraint_violation = compute_constraint_violation(A)

        if verbose:
            print(f"Found good constraint weighting: {constraint_weighting}.")

    # get callback for stop criterion an status messages
    callback = get_callback(
        problem=problem,
        model_type=model_formula,
        n_max_min_found=n_max_min_found,
        n_max_no_change=n_max_no_change,
        tol=tol,
        verbose=verbose,
    )

    # get objective function
    objective = get_objective(problem, model_type, tol, constraint_weighting)

    result = basinhopping(
        objective,
        x0=problem.sample_inputs(n_experiments).values.reshape(-1),
        minimizer_kwargs={
            "method": "SLSQP",
            "bounds": [(p.bounds) for p in problem.inputs] * n_experiments,
        },
        callback=callback,
    )

    A = pd.DataFrame(
        result["x"].reshape(n_experiments, D), columns=problem.inputs.names
    )
    A.index = [f"exp{i}" for i in range(len(A))]
    return A


# Test problem 1 from paper
problem = opti.Problem(
    inputs=opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(3)]),
    outputs=[opti.Continuous("y")],
    constraints=[
        opti.LinearEquality(names=["x0", "x1", "x2"], rhs=1),
        opti.LinearInequality(["x1"], lhs=[-1], rhs=-0.1),
        opti.LinearInequality(["x2"], lhs=[1], rhs=0.6),
        opti.LinearInequality(["x0", "x1"], lhs=[5, 4], rhs=3.9),
        opti.LinearInequality(["x0", "x1"], lhs=[-20, 5], rhs=-3),
    ],
)

# import warnings
# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    print(optimal_design(problem, "linear", n_experiments=12, verbose=True, ctol=1e-4))

# Test problem 2 from paper
problem = opti.Problem(
    inputs=opti.Parameters(
        [
            opti.Continuous("x1", domain=[0.2, 0.65]),
            opti.Continuous("x2", domain=[0.1, 0.55]),
            opti.Continuous("x3", domain=[0.1, 0.2]),
            opti.Continuous("x4", domain=[0.15, 0.35]),
        ]
    ),
    outputs=[opti.Continuous("y")],
    constraints=[opti.LinearEquality(names=["x1", "x2", "x3", "x4"], rhs=1)],
)


# TODO:
# tests schreiben
# constraints in SLSQP einpfelgen
# Sampling mit NChooseK Constraint
# bounds im accept test implementieren
# bessere Stoppkriterien werden gebraucht
# Strategie für richtiges Verhältnis penalty/obj?
#       fixen, möglichst guten Zusammenhang finden
#       anfangen bei Skalierungsfaktor 1 für penalty --> optimieren
#           --> constraintsverletzung zu groß? --> erneut mit größerem Skalierungsfaktor --> ...
#       anfangen mit Skalierungsfaktor 1 für penalty --> kurz optimieren
#           --> aus Verhältnis von logD und penalty für Bestimmung des Skalierungsfaktors nutzen
# pymoo ausprobieren


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
