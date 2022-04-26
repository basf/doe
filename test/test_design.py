import numpy as np
import opti

from doe.design import find_local_max_ipopt, get_objective, logD
from doe.utils import get_formula_from_string, n_zero_eigvals


def test_logD():
    A = np.ones(shape=(10, 5))
    A[0, 0] = 2

    assert np.allclose(logD(A), np.linalg.slogdet(A.T @ A + 1e-7 * np.eye(5))[1])


def test_get_objective():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )
    objective = get_objective(problem, "linear")

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(objective(x), -np.log(4) - np.log(1e-7))


def test_find_local_max_ipopt_nchoosek():
    # Design for a problem with an n-choose-k constraint
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=3)],
    )
    D = problem.n_inputs
    N = (
        len(get_formula_from_string(problem=problem, model_type="linear").terms)
        - n_zero_eigvals(problem=problem, model_type="linear")
        + 3
    )
    A = find_local_max_ipopt(problem, "linear")
    assert A.shape == (N, D)


def test_find_local_max_ipopt_mixture():
    # Design for a problem with a mixture constraint
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(inputs.names, rhs=1)],
    )
    D = problem.n_inputs
    N = (
        len(get_formula_from_string(problem=problem, model_type="linear").terms)
        - n_zero_eigvals(problem=problem, model_type="linear")
        + 3
    )
    A = find_local_max_ipopt(problem, "linear")
    assert A.shape == (N, D)


def test_find_local_max_ipopt_results():
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

    np.random.seed(1)
    A = find_local_max_ipopt(problem, "linear", n_experiments=12)
    opt = np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]])
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=5e-3) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=5e-3) for row in A.to_numpy()])
