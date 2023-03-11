import sys

import numpy as np
import opti
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint

from doe.utils import (
    ConstraintWrapper,
    ProblemContext,
    a_optimality,
    check_nchoosek_constraints_as_bounds,
    constraints_as_scipy_constraints,
    d_optimality,
    g_efficiency,
    get_formula_from_string,
    metrics,
    n_zero_eigvals,
    nchoosek_constraints_as_bounds,
)


def test_get_formula_from_string_recursion_limit():
    # save recursion limit
    recursion_limit = sys.getrecursionlimit()

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model_type=model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert model.terms[i] in terms
        assert terms[i] in model.terms

    assert recursion_limit == sys.getrecursionlimit()


def test_get_formula_from_string():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )

    problem_context = ProblemContext(problem)

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0**2", "x1**2", "x2**2"]
    model_formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # fully quadratic
    terms = [
        "1",
        "x0",
        "x1",
        "x2",
        "x0:x1",
        "x0:x2",
        "x1:x2",
        "x0**2",
        "x1**2",
        "x2**2",
    ]
    model_formula = get_formula_from_string(
        problem_context=problem_context, model_type="fully-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0**2", "x0:x1"]
    model_formula = get_formula_from_string(
        problem_context=problem_context,
        model_type="y ~ 1 + x0 + x0:x1 + {x0**2}",
        rhs_only=False,
    )
    assert all(term in terms_lhs for term in model_formula.terms.lhs)
    assert all(term in model_formula.terms.lhs for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.terms.rhs)
    assert all(term in model_formula.terms.rhs for term in terms_rhs)

    # get formula without model: valid input
    model = "x1 + x2 + x3"
    model = get_formula_from_string(model_type=model)
    assert str(model) == "1 + x1 + x2 + x3"

    # get formula without model: invalid input
    with pytest.raises(AssertionError):
        model = get_formula_from_string("linear")

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model_type=model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert model.terms[i] in terms
        assert terms[i] in model.terms


def test_formula_from_string_with_categoricals():
    d = 2
    inputs = [opti.Categorical(f"x{i+1}", ["a", "b", "c"]) for i in range(d)]
    inputs.append(opti.Continuous(f"x{4}", [0, 1]))
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
    )
    problem_context = ProblemContext(problem)

    model_formula = problem_context.get_formula_from_string(model_type="linear")
    # linear and interaction
    terms = [
        "1",
        "x1",
        "x2",
        "x4",
    ]
    model_formula = problem_context.get_formula_from_string(model_type="linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1",
        "x2",
        "x4",
        "x4**2",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="linear-and-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1",
        "x2",
        "x4",
        "x1:x4",
        "x2:x4",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1",
        "x2",
        "x4",
        "x4**2",
        "x1:x4",
        "x2:x4",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="fully-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    problem_context = ProblemContext(problem)
    problem_context.relax_problem()

    model_formula = problem_context.get_formula_from_string(model_type="linear")
    # linear and interaction
    terms = [
        "1",
        "x1____a",
        "x1____b",
        "x1____c",
        "x2____a",
        "x2____b",
        "x2____c",
        "x4",
    ]
    model_formula = problem_context.get_formula_from_string(model_type="linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1____a",
        "x1____b",
        "x1____c",
        "x2____a",
        "x2____b",
        "x2____c",
        "x4",
        "x4**2",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="linear-and-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1____a",
        "x1____b",
        "x1____c",
        "x2____a",
        "x2____b",
        "x2____c",
        "x4",
        "x1____a:x4",
        "x1____b:x4",
        "x1____c:x4",
        "x2____a:x4",
        "x2____b:x4",
        "x2____c:x4",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    terms = [
        "1",
        "x1____a",
        "x1____b",
        "x1____c",
        "x2____a",
        "x2____b",
        "x2____c",
        "x4",
        "x1____a:x4",
        "x1____b:x4",
        "x1____c:x4",
        "x2____a:x4",
        "x2____b:x4",
        "x2____c:x4",
        "x4**2",
    ]
    model_formula = problem_context.get_formula_from_string(
        model_type="fully-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)


def test_has_constraint_with_discrete_or_categorical():
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1", domain=[0, 100]),
            opti.Continuous("x2", domain=[0, 100]),
            opti.Continuous("x3", domain=[0, 100]),
            opti.Discrete("discrete1", [0, 1, 5]),
            opti.Discrete("discrete2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x1", "x2", "discrete1"], rhs=1)],
    )
    problem_context = ProblemContext(problem=prob)
    assert problem_context.has_constraint_with_cats_or_discrete_variables

    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1", domain=[0, 100]),
            opti.Continuous("x2", domain=[0, 100]),
            opti.Continuous("x3", domain=[0, 100]),
            opti.Categorical("cat1", ["a", "b"]),
            opti.Categorical("cat2", ["a", "b"]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["cat1"], rhs="a")],
    )
    problem_context = ProblemContext(problem=prob)
    assert problem_context.has_constraint_with_cats_or_discrete_variables


def test_n_zero_eigvals_unconstrained():
    # 5 continous & 1 categorical inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 100]) for i in range(5)],
        outputs=[opti.Continuous("y")],
    )
    problem_context = ProblemContext(problem)
    assert n_zero_eigvals(problem_context=problem_context, model_type="linear") == 0
    assert (
        n_zero_eigvals(
            problem_context=problem_context, model_type="linear-and-quadratic"
        )
        == 0
    )
    assert (
        n_zero_eigvals(
            problem_context=problem_context, model_type="linear-and-interactions"
        )
        == 0
    )
    assert (
        n_zero_eigvals(problem_context=problem_context, model_type="fully-quadratic")
        == 0
    )


def test_n_zero_eigvals_constrained():
    # 3 continuous & 2 discrete inputs, 1 mixture constraint
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1", domain=[0, 100]),
            opti.Continuous("x2", domain=[0, 100]),
            opti.Continuous("x3", domain=[0, 100]),
            opti.Discrete("discrete1", [0, 1, 5]),
            opti.Discrete("discrete2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x1", "x2", "x3"], rhs=1)],
    )
    problem_context = ProblemContext(problem=prob)
    assert n_zero_eigvals(problem_context, "linear") == 1
    assert n_zero_eigvals(problem_context, "linear-and-quadratic") == 1
    assert n_zero_eigvals(problem_context, "linear-and-interactions") == 3
    assert n_zero_eigvals(problem_context, "fully-quadratic") == 6

    # TODO: NChooseK?


def test_number_of_model_terms():
    # 5 continous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    problem_context = ProblemContext(problem)

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear"
    )
    assert len(formula.terms) == 6

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-quadratic"
    )
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="fully-quadratic"
    )
    assert len(formula.terms) == 21

    # 3 continuous & 2 discrete inputs
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", domain=[0, 100]),
            opti.Continuous("x2", domain=[0, 100]),
            opti.Continuous("x3", domain=[0, 100]),
            opti.Discrete("discrete1", [0, 1, 5]),
            opti.Discrete("discrete2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
    )

    problem_context = ProblemContext(problem)

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear"
    )
    assert len(formula.terms) == 6

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-quadratic"
    )
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(
        problem_context=problem_context, model_type="fully-quadratic"
    )
    assert len(formula.terms) == 21


def test_constraints_as_scipy_constraints():
    # test problems from the paper "The construction of D- and I-optimal designs for
    # mixture experiments with linear constraints on the components" by R. Coetzer and
    # L. M. Haines.

    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3"], rhs=1),
            opti.LinearInequality(["x2"], lhs=[-1], rhs=-0.1),
            opti.LinearInequality(["x3"], lhs=[1], rhs=0.6),
            opti.LinearInequality(["x1", "x2"], lhs=[5, 4], rhs=3.9),
            opti.LinearInequality(["x1", "x2"], lhs=[-20, 5], rhs=-3),
        ],
    )

    n_experiments = 2

    constraints = constraints_as_scipy_constraints(problem, n_experiments)

    for c in constraints:
        assert isinstance(c, LinearConstraint)
        assert c.A.shape == (n_experiments, problem.n_inputs * n_experiments)
        assert len(c.lb) == n_experiments
        assert len(c.ub) == n_experiments

    A = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]) / np.sqrt(3)
    lb = np.array([1, 1]) / np.sqrt(3) - 0.001
    ub = np.array([1, 1]) / np.sqrt(3) + 0.001
    assert np.allclose(constraints[0].A, A)
    assert np.allclose(constraints[0].lb, lb)
    assert np.allclose(constraints[0].ub, ub)

    lb = -np.inf * np.ones(n_experiments)
    ub = -0.1 * np.ones(n_experiments)
    assert np.allclose(constraints[1].lb, lb)
    assert np.allclose(constraints[1].ub, ub)

    # problem with nonlinear constraints
    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NonlinearEquality("x1**2 + x2**2 - 1"),
            opti.NonlinearInequality("x1**2 + x2**2 - 1"),
        ],
    )

    constraints = constraints_as_scipy_constraints(problem, n_experiments)

    for c in constraints:
        assert isinstance(c, NonlinearConstraint)
        assert len(c.lb) == n_experiments
        assert len(c.ub) == n_experiments
        assert np.allclose(c.fun(np.array([1, 1, 1, 1, 1, 1])), [1, 1])

    # problem with NChooseK constraints: ignore
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [-1, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=2)],
    )
    n_experiments = 1

    constraints = constraints_as_scipy_constraints(problem, n_experiments)
    assert len(constraints) == 0


def test_ConstraintWrapper():
    # define problem with all types of constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.LinearInequality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.NonlinearEquality(
                "x1**2 + x2**2 + x3**2 + x4**2 - 1", jacobian="[2*x1,2*x2,2*x3,2*x4]"
            ),
            opti.NonlinearInequality(
                "x1**2 + x2**2 + x3**2 + x4**2 - 1", jacobian="[2*x1,2*x2,2*x3,2*x4]"
            ),
            opti.NChooseK(names=["x1", "x2", "x3", "x4"], max_active=3),
        ],
    )

    x = np.array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [3, 2, 1, 0]]).flatten()

    # linear equality
    c = ConstraintWrapper(problem.constraints[0], problem, n_experiments=3, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))
    assert np.allclose(
        c.jacobian(x),
        np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ]
        ),
    )

    # linear inequaity
    c = ConstraintWrapper(problem.constraints[1], problem, n_experiments=3, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))
    assert np.allclose(
        c.jacobian(x),
        np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ]
        ),
    )

    # nonlinear equality
    c = ConstraintWrapper(problem.constraints[2], problem, n_experiments=3, tol=0)
    assert np.allclose(c(x), np.array([3, 0, 13]))
    assert np.allclose(
        c.jacobian(x),
        2
        * np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0],
            ]
        ),
    )

    # nonlinear inequality
    c = ConstraintWrapper(problem.constraints[3], problem, n_experiments=3, tol=0)
    assert np.allclose(c(x), np.array([3, 0, 13]))
    assert np.allclose(
        c.jacobian(x),
        2
        * np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0],
            ]
        ),
    )

    # nchoosek constraint
    c = ConstraintWrapper(problem.constraints[4], problem, n_experiments=3, tol=0)
    assert np.allclose(c(x), np.array([1, 0.5, 0]))


def test_d_optimality():
    # define model matrix: full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    assert np.allclose(d_optimality(X), np.linalg.slogdet(X.T @ X)[1])

    # define model matrix: not full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert np.allclose(d_optimality(X), np.sum(np.log(np.linalg.eigvalsh(X.T @ X)[1:])))


def test_a_optimality():
    # define model matrix: full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    assert np.allclose(a_optimality(X), np.sum(1 / (np.linalg.eigvalsh(X.T @ X))))

    # define model matrix: not full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert np.allclose(a_optimality(X), np.sum(1 / (np.linalg.eigvalsh(X.T @ X)[1:])))


def test_g_efficiency():
    # define model matrix and problem: no constraints
    X = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ]
    )

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0.95, 1.0]) for i in range(4)],
        outputs=[opti.Continuous("y")],
    )
    assert np.allclose(g_efficiency(X, problem), 0.333, atol=5e-3)

    # define problem: sampling not implemented
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0.95, 1.0]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3", "x4"], max_active=1),
        ],
    )
    with pytest.raises(Exception):
        g_efficiency(X, problem, n_samples=1)


def test_metrics():
    # define model matrix
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0.95, 1.0]) for i in range(4)],
        outputs=[opti.Continuous("y")],
    )

    d = metrics(X, problem)
    assert d.index[0] == "D-optimality"
    assert d.index[1] == "A-optimality"
    assert d.index[2] == "G-efficiency"
    assert np.allclose(
        d,
        np.array([d_optimality(X), a_optimality(X), g_efficiency(X, problem)]),
        rtol=0.05,
    )

    # define problem: sampling not implemented
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0.95, 1.0]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3", "x4"], max_active=1),
        ],
    )
    with pytest.warns(UserWarning):
        d = metrics(X, problem, n_samples=1)
    assert d.index[0] == "D-optimality"
    assert d.index[1] == "A-optimality"
    assert d.index[2] == "G-efficiency"
    assert np.allclose(d, np.array([d_optimality(X), a_optimality(X), 0]))


def test_check_nchoosek_constraints_as_bounds():
    # define problem: possible to formulate as bounds, no NChooseK constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
    )
    check_nchoosek_constraints_as_bounds(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-1, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[],
    )
    check_nchoosek_constraints_as_bounds(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [None, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=["x1", "x2"])],
    )
    check_nchoosek_constraints_as_bounds(problem)

    # define problem: possible to formulate as bounds, with NChooseK and other constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-i, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2"]),
            opti.LinearInequality(names=["x3", "x4"]),
            opti.NChooseK(names=["x1", "x2"], max_active=1),
            opti.NChooseK(names=["x3", "x4"], max_active=1),
        ],
    )
    check_nchoosek_constraints_as_bounds(problem)

    # define problem: not possible to formulate as bounds, invalid bounds
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0.1, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_as_bounds(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-1, -0.1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_as_bounds(problem)

    # define problem: not possible to formulate as bounds, names parameters of two NChooseK overlap
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
            opti.NChooseK(names=["x2", "x3", "x4"], max_active=2),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_as_bounds(problem)


def test_nchoosek_constraints_as_bounds():
    # define problem: no NChooseK constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-1, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
    )
    bounds = nchoosek_constraints_as_bounds(problem, n_experiments=4)
    assert len(bounds) == 20
    _bounds = [p.bounds for p in problem.inputs] * 4
    for i in range(20):
        assert _bounds[i] == bounds[i]

    # define problem: with NChooseK constraints
    # define problem: no NChooseK constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-1, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(["x1", "x2", "x3"], max_active=1)],
    )
    np.random.seed(1)
    bounds = nchoosek_constraints_as_bounds(problem, n_experiments=4)
    _bounds = [
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    assert len(bounds) == 20
    for i in range(20):
        assert _bounds[i] == bounds[i]


def test_ProblemContext_has_constraint_with_cats_or_discrete_variables():
    # No constriants
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1])],
        outputs=[opti.Continuous("y")],
    )
    problem_context = ProblemContext(problem)
    assert not problem_context.has_constraint_with_cats_or_discrete_variables

    # Constraints not on categorical or discrete variables
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Categorical("x2", ["A", "B"]),
            opti.Discrete("x3", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=["x1"])],
    )
    problem_context = ProblemContext(problem)
    assert not problem_context.has_constraint_with_cats_or_discrete_variables

    # Nonlinear Constraints
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Categorical("x2", ["A", "B"]),
            opti.Discrete("x3", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.NonlinearEquality("x2**2 + x3**2")],
    )
    problem_context = ProblemContext(problem)
    assert not problem_context.has_constraint_with_cats_or_discrete_variables

    # Constraints on categorical/discrete variables
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Categorical("x2", ["A", "B"]),
            opti.Discrete("x3", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=["x1", "x2", "x3"])],
    )
    problem_context = ProblemContext(problem)
    assert problem_context.has_constraint_with_cats_or_discrete_variables
