import numpy as np
import opti
import pandas as pd

from doe.design import (
    get_callback,
    get_constraint_violation,
    get_formula_from_string,
    get_objective,
    logD_,
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
    callback = get_callback(
        problem, "linear", n_max_min_found=1, verbose=False, tol=1e-3
    )
    x = np.array([-1, 1, 1, 1])
    assert callback(x, 1, False)

    callback = get_callback(
        problem, "linear", n_max_no_change=1, verbose=False, tol=1e-3
    )
    assert callback(x, 1, False)

    assert callback.n_calls == 1
    assert callback.f_opt == 1
    assert callback.cv_opt == 0.999
    assert not callback.verbose
    assert callback.n_no_change == 1


def test_get_constraint_violation():
    # linear equality constraint
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x0"], lhs=1, rhs=0)],
    )
    constraint_violation = get_constraint_violation(problem)

    x = pd.DataFrame([[-1, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0.999)
    x = pd.DataFrame([[-5e-4, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)
    x = pd.DataFrame([[1, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0.999)
    x = pd.DataFrame([[5e-4, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)

    # linear inequality constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearInequality(["x0"], lhs=1, rhs=0)],
    )
    constraint_violation = get_constraint_violation(problem)

    x = pd.DataFrame([[-1, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)
    x = pd.DataFrame([[1, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0.999)
    x = pd.DataFrame([[5e-4, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)

    # NChooseK constraint
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(5)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(["x0", "x1", "x2", "x3", "x4"], 4)],
    )
    constraint_violation = get_constraint_violation(problem)

    x = pd.DataFrame([[-0.5, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0.499)
    x = pd.DataFrame([[-5e-4, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)
    x = pd.DataFrame([[0.5, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0.499)
    x = pd.DataFrame([[5e-4, 1, 1, 1, 1]], columns=problem.inputs.names)
    assert np.allclose(constraint_violation(x), 0)


def test_get_objective_unconstrained():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )
    objective = get_objective(problem, "linear", tol=0)

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(objective(x), 35.35050620855721)


def test_get_objective_constrained():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x0", "x1", "x2"], lhs=1, rhs=1)],
    )
    objective = get_objective(problem, "linear", tol=0)

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(objective(x), -np.log(4))

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(2)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x0", "x1"], lhs=1, rhs=1)],
    )
    objective = get_objective(problem, "x0 + x1 -1", tol=0, constraint_weight=1)
    x = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert np.allclose(objective(x), 1)


def test_logD_():
    A = np.ones(shape=(10, 5))
    A[0, 0] = 2
    assert np.allclose(logD_(A, 3), np.log(5.23118190e01) + np.log(6.88181002e-01))


def test_get_n_ignore_eigvals_unconstrained():
    # 5 continous & 1 categorical inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)]
        + [opti.Categorical("x5", ["A", "B"])],
        outputs=[opti.Continuous("y")],
    )

    assert n_ignore_eigvals(problem, "linear", 0) == 0
    assert n_ignore_eigvals(problem, "linear-and-quadratic", 0) == 0
    assert n_ignore_eigvals(problem, "linear-and-interactions", 0) == 0
    assert n_ignore_eigvals(problem, "fully-quadratic", 0) == 0


def test_get_n_ignore_eigvals_constrained():
    # 3 continuous & 2 discrete inputs, 1 mixture constraint
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Continuous("x3"),
            opti.Discrete("discrete1", [0, 1, 5]),
            opti.Discrete("discrete2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x1", "x2", "x3"], rhs=1)],
    )

    assert n_ignore_eigvals(prob, "linear", 0) == 1
    assert n_ignore_eigvals(prob, "linear-and-quadratic", 0) == 1
    assert n_ignore_eigvals(prob, "linear-and-interactions", 0) == 3
    assert n_ignore_eigvals(prob, "fully-quadratic", 0) == 6

    # NChooseK?


def test_number_of_model_terms():
    # 5 continous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    assert len(get_formula_from_string(problem, "linear").terms) == 6
    assert len(get_formula_from_string(problem, "linear").terms) == 11
    assert len(get_formula_from_string(problem, "linear").terms) == 16
    assert len(get_formula_from_string(problem, "linear").terms) == 21

    # 3 continuous & 2 discrete inputs
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Continuous("x3"),
            opti.Discrete("discrete1", [0, 1, 5]),
            opti.Discrete("discrete2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
    )

    assert n_ignore_eigvals(problem, "linear", 0) == 6
    assert n_ignore_eigvals(problem, "linear-and-quadratic", 0) == 11
    assert n_ignore_eigvals(problem, "linear-and-interactions", 0) == 16
    assert n_ignore_eigvals(problem, "fully-quadratic", 0) == 21


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
    A = optimal_design(problem)
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
    A = optimal_design(problem)
    assert A.shape == (N, D)
