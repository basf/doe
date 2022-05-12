import numpy as np
import opti
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint

from doe.utils import (
    ConstraintWrapper,
    JacobianNChooseK,
    a_optimality,
    check_nchoosek_constraints_linearizable,
    constraints_as_scipy_constraints,
    d_optimality,
    g_efficiency,
    get_formula_from_string,
    metrics,
    n_zero_eigvals,
    nchoosek_constraint_as_scipy_linear_constraint,
)


def test_get_formula_from_string():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(problem=problem, model_type="linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(
        problem=problem, model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0**2", "x1**2", "x2**2"]
    model_formula = get_formula_from_string(
        problem=problem, model_type="linear-and-quadratic"
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
        problem=problem, model_type="fully-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0**2", "x0:x1"]
    model_formula = get_formula_from_string(
        problem=problem, model_type="y ~ 1 + x0 + x0:x1 + {x0**2}", rhs_only=False
    )
    assert all(term in terms_lhs for term in model_formula.terms.lhs)
    assert all(term in model_formula.terms.lhs for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.terms.rhs)
    assert all(term in model_formula.terms.rhs for term in terms_rhs)

    # get formula without model: valid input
    model = "x1 + x2 + x3"
    model = get_formula_from_string(model)
    assert str(model) == "1 + x1 + x2 + x3"

    # get formula without model: invalid input
    with pytest.raises(AssertionError):
        model = get_formula_from_string("linear")

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert model.terms[i] in terms
        assert terms[i] in model.terms


def test_n_zero_eigvals_unconstrained():
    # 5 continous & 1 categorical inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 100]) for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    assert n_zero_eigvals(problem, "linear") == 0
    assert n_zero_eigvals(problem, "linear-and-quadratic") == 0
    assert n_zero_eigvals(problem, "linear-and-interactions") == 0
    assert n_zero_eigvals(problem, "fully-quadratic") == 0


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

    assert n_zero_eigvals(prob, "linear") == 1
    assert n_zero_eigvals(prob, "linear-and-quadratic") == 1
    assert n_zero_eigvals(prob, "linear-and-interactions") == 3
    assert n_zero_eigvals(prob, "fully-quadratic") == 6

    # TODO: NChooseK?


def test_number_of_model_terms():
    # 5 continous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    formula = get_formula_from_string(problem=problem, model_type="linear")
    assert len(formula.terms) == 6

    formula = get_formula_from_string(
        problem=problem, model_type="linear-and-quadratic"
    )
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        problem=problem, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(problem=problem, model_type="fully-quadratic")
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

    formula = get_formula_from_string(problem=problem, model_type="linear")
    assert len(formula.terms) == 6

    formula = get_formula_from_string(
        problem=problem, model_type="linear-and-quadratic"
    )
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        problem=problem, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(problem=problem, model_type="fully-quadratic")
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

    # problem with NChooseK constraint: no linearization
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=2)],
    )
    n_experiments = 5

    x = np.array(
        [
            [1, -10, 2, -1.5],
            [2, -10, 3, 5],
            [0, 1, 0, -2],
            [2, -1, 1e-5, 1],
            [1, 1, 1, 1],
        ]
    ).flatten()

    constraints = constraints_as_scipy_constraints(problem, n_experiments)

    assert isinstance(constraints[0], NonlinearConstraint)
    assert len(constraints[0].lb) == n_experiments
    assert len(constraints[0].ub) == n_experiments
    assert np.allclose(
        constraints[0].fun(x),
        problem.constraints[0](
            pd.DataFrame(x.reshape(5, 4), columns=["x0", "x1", "x2", "x3"])
        ),
    )

    # problem with NChooseK constraint: with linearization
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=2)],
    )
    n_experiments = 7

    np.random.seed(1)
    with pytest.warns(UserWarning):
        constraints = constraints_as_scipy_constraints(
            problem, n_experiments, linearize_NChooseK=True
        )

    A = np.array(
        [
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
        ]
    ) / np.sqrt(14)
    assert isinstance(constraints[0], LinearConstraint)
    assert np.allclose(constraints[0].A, A)
    assert np.allclose(constraints[0].ub, [1e-3])
    assert np.allclose(constraints[0].lb, [-np.inf])

    # problem with NChooseK constraint: with linearization, not linearizable
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [-1, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=2)],
    )
    n_experiments = 1

    with pytest.raises(ValueError):
        constraints = constraints_as_scipy_constraints(
            problem, n_experiments, linearize_NChooseK=True
        )


def test_JacobianNChooseK():

    # problem with NChooseK constraint
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=2)],
    )
    n_experiments = 5
    D = problem.n_inputs

    x = np.array(
        [
            [1, -10, 2, -1.5],
            [2, -10, 3, 5],
            [0, 1, 0, -2],
            [2, -1, 1e-5, 1],
            [1, 1, 1, 1],
        ]
    ).flatten()

    jac = JacobianNChooseK(problem.constraints[0], problem, n_experiments)

    np.random.seed(1)
    jac_corr = np.array(
        [
            [1, 0, 0, -1],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    J = np.zeros(shape=(n_experiments, D * n_experiments))
    for i in range(n_experiments):
        J[i, i * D : (i + 1) * D] = jac_corr[i]
    assert np.allclose(jac(x), J)

    np.random.seed(100)
    jac_corr = np.array(
        [
            [1, 0, 0, -1],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
        ]
    )
    J = np.zeros(shape=(n_experiments, D * n_experiments))
    for i in range(n_experiments):
        J[i, i * D : (i + 1) * D] = jac_corr[i]
    assert np.allclose(jac(x), J)

    np.random.seed(200)
    jac_corr = np.array(
        [
            [1, 0, 0, -1],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 1, 0],
        ]
    )
    J = np.zeros(shape=(n_experiments, D * n_experiments))
    for i in range(n_experiments):
        J[i, i * D : (i + 1) * D] = jac_corr[i]
    assert np.allclose(jac(x), J)


def test_ConstraintWrapper():
    # define problem with all types of constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.LinearInequality(names=["x1", "x2", "x3", "x4"], rhs=1),
            opti.NonlinearEquality("x1**2 + x2**2 + x3**2 + x4**2 - 1"),
            opti.NonlinearInequality("x1**2 + x2**2 + x3**2 + x4**2 - 1"),
            opti.NChooseK(names=["x1", "x2", "x3", "x4"], max_active=3),
        ],
    )

    x = np.array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [3, 2, 1, 0]]).flatten()

    # linear equality
    c = ConstraintWrapper(problem.constraints[0], problem, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))

    # linear inequaity
    c = ConstraintWrapper(problem.constraints[1], problem, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))

    # nonlinear equality
    c = ConstraintWrapper(problem.constraints[2], problem, tol=0)
    assert np.allclose(c(x), np.array([3, 0, 13]))

    # nonlinear inequality
    c = ConstraintWrapper(problem.constraints[3], problem, tol=0)
    assert np.allclose(c(x), np.array([3, 0, 13]))

    # nchoosek constraint
    c = ConstraintWrapper(problem.constraints[4], problem, tol=0)
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


def test_check_nchoosek_constraints_linearizable():
    # define problem: linearizable, no NChooseK constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
    )
    check_nchoosek_constraints_linearizable(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[],
    )
    check_nchoosek_constraints_linearizable(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=["x1", "x2"])],
    )
    check_nchoosek_constraints_linearizable(problem)

    # define problem: linearizable, with NChooseK and other constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2"]),
            opti.LinearInequality(names=["x3", "x4"]),
            opti.NChooseK(names=["x1", "x2"], max_active=1),
            opti.NChooseK(names=["x3", "x4"], max_active=1),
        ],
    )
    check_nchoosek_constraints_linearizable(problem)

    # define problem: not linearizable, invalid bounds
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [None, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_linearizable(problem)

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [-i, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_linearizable(problem)

    # define problem: not linearizable, names parameters of two NChooseK overlap
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x1", "x2"], max_active=1),
            opti.NChooseK(names=["x2", "x3", "x4"], max_active=2),
        ],
    )
    with pytest.raises(ValueError):
        check_nchoosek_constraints_linearizable(problem)


def test_nchoosek_as_scipy_linear_constraint():
    # define constraints and problem names: standard case
    constraint = opti.NChooseK(names=["x1", "x2", "x3", "x4"], max_active=3)
    names = [f"x{i+1}" for i in range(6)]
    n_experiments = 10

    linear_constraint = nchoosek_constraint_as_scipy_linear_constraint(
        constraint,
        names,
        n_experiments,
    )

    assert isinstance(linear_constraint, LinearConstraint)
    assert np.shape(linear_constraint.A) == (1, len(names) * n_experiments)
    assert np.allclose(linear_constraint.ub, [1e-3])
    assert np.allclose(linear_constraint.lb, [-np.inf])

    # define constraints and problem names: standard case, assert all entries are correct
    constraint = opti.NChooseK(names=["x1", "x3", "x4"], max_active=1)
    names = [f"x{i+1}" for i in range(4)]
    n_experiments = 4

    np.random.seed(1)
    linear_constraint = nchoosek_constraint_as_scipy_linear_constraint(
        constraint,
        names,
        n_experiments,
    )

    A = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0]) / np.sqrt(8)

    assert np.allclose(linear_constraint.A, A)
    assert np.allclose(linear_constraint.ub, [1e-3])
    assert np.allclose(linear_constraint.lb, [-np.inf])
