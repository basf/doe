import numpy as np
import opti
from scipy.optimize import LinearConstraint, NonlinearConstraint

from doe.design import (
    constraints_as_scipy_constraints,
    get_callback,
    get_formula_from_string,
    get_objective,
    logD,
    n_ignore_eigvals,
    optimal_design,
)


def test_get_formula_from_string():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(problem, "linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(problem, "linear-and-interactions")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0**2", "x1**2", "x2**2"]
    model_formula = get_formula_from_string(problem, "linear-and-quadratic")
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
    model_formula = get_formula_from_string(problem, "fully-quadratic")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0**2", "x0:x1"]
    model_formula = get_formula_from_string(
        problem, "y ~ 1 + x0 + x0:x1 + {x0**2}", rhs_only=False
    )
    assert all(term in terms_lhs for term in model_formula.terms.lhs)
    assert all(term in model_formula.terms.lhs for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.terms.rhs)
    assert all(term in model_formula.terms.rhs for term in terms_rhs)


def test_get_callback():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(4)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x0", "x1", "x2", "x3"], lhs=1, rhs=0)],
    )

    # test stop criteria
    callback = get_callback(problem, n_max_min_found=1, verbose=False, tol=1e-3)
    x = np.array([-1, 1, 1, 1])
    assert callback(x, 1, False)

    callback = get_callback(problem, n_max_no_change=1, verbose=False, tol=1e-3)
    assert callback(x, 1, False)

    assert callback.n_calls == 1
    assert callback.f_opt == 1
    assert callback.cv_opt == 0.999
    assert not callback.verbose
    assert callback.n_no_change == 1


def test_get_callback_constraint_violation():
    # linear equality constraint
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x0"], lhs=1, rhs=0)],
    )

    callback = get_callback(problem, tol=1e-3)
    x = np.array([-1, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0.999)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([-5e-4, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([1, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0.999)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([5e-4, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0)

    # linear inequality constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearInequality(["x0"], lhs=1, rhs=0)],
    )

    callback = get_callback(problem, tol=1e-3)
    x = np.array([-1, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([1, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 1)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([5e-4, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0.0005)

    # NChooseK constraint
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(["x0", "x1", "x2", "x3", "x4"], 4)],
    )

    callback = get_callback(problem, tol=1e-3)
    x = np.array([-0.5, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0.499)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([-5e-4, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([0.5, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0.499)

    callback = get_callback(problem, tol=1e-3)
    x = np.array([5e-4, 1, 1, 1, 1])
    callback(x, 1, False)
    assert np.allclose(callback.cv_opt, 0)


def test_get_objective():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )
    objective = get_objective(problem, "linear")

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(objective(x), 35.35050620855721)


def test_logD():
    A = np.ones(shape=(10, 5))
    A[0, 0] = 2
    assert np.allclose(logD(A, 3), np.log(5.23118190e01) + np.log(6.88181002e-01))


def test_n_ignore_eigvals_unconstrained():
    # 5 continous & 1 categorical inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 100]) for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    assert n_ignore_eigvals(problem, "linear") == 0
    assert n_ignore_eigvals(problem, "linear-and-quadratic") == 0
    assert n_ignore_eigvals(problem, "linear-and-interactions") == 0
    assert n_ignore_eigvals(problem, "fully-quadratic") == 0


def test_n_ignore_eigvals_constrained():
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

    assert n_ignore_eigvals(prob, "linear") == 1
    assert n_ignore_eigvals(prob, "linear-and-quadratic") == 1
    assert n_ignore_eigvals(prob, "linear-and-interactions") == 3
    assert n_ignore_eigvals(prob, "fully-quadratic") == 6

    # NChooseK?


def test_number_of_model_terms():
    # 5 continous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    assert len(get_formula_from_string(problem, "linear").terms) == 6
    assert len(get_formula_from_string(problem, "linear-and-quadratic").terms) == 11
    assert len(get_formula_from_string(problem, "linear-and-interactions").terms) == 16
    assert len(get_formula_from_string(problem, "fully-quadratic").terms) == 21

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

    assert len(get_formula_from_string(problem, "linear").terms) == 6
    assert len(get_formula_from_string(problem, "linear-and-quadratic").terms) == 11
    assert len(get_formula_from_string(problem, "linear-and-interactions").terms) == 16
    assert len(get_formula_from_string(problem, "fully-quadratic").terms) == 21


def test_optimal_design_nchoosek():
    # Design for a problem with an n-choose-k constraint
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=3)],
    )
    D = problem.n_inputs
    N = (
        len(get_formula_from_string(problem, "linear").terms)
        - n_ignore_eigvals(problem, "linear")
        + 3
    )
    A = optimal_design(problem, "linear")
    assert A.shape == (N, D)


def test_optimal_design_mixture():
    # Design for a problem with a mixture constraint
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(inputs.names, rhs=1)],
    )
    D = problem.n_inputs
    N = (
        len(get_formula_from_string(problem, "linear").terms)
        - n_ignore_eigvals(problem, "linear")
        + 3
    )
    A = optimal_design(problem, "linear")
    assert A.shape == (N, D)


def test_optimal_design_results():
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

    A = optimal_design(problem, "linear", n_experiments=12)
    opt = np.array([[0.3, 0.1, 0.6], [0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2]])
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=5e-3) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=5e-3) for row in A.to_numpy()])


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

    A = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    lb = np.array([0.999, 0.999])
    ub = np.array([1.001, 1.001])
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
