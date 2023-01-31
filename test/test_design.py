import warnings

import numpy as np
import opti
import pytest

from doe.design import (
    check_constraints_and_domain_respected,
    check_fixed_experiments,
    find_local_max_ipopt,
    get_objective,
    logD,
)
from doe.sampling import CornerSampling, OptiSampling, ProbabilitySimplexSampling
from doe.utils import ProblemContext, get_formula_from_string, n_zero_eigvals


def test_logD():
    A = np.ones(shape=(10, 5))
    A[0, 0] = 2

    assert np.allclose(logD(A), np.linalg.slogdet(A.T @ A + 1e-7 * np.eye(5))[1])


def test_get_objective():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )
    problem_context = ProblemContext(problem=problem)
    objective = get_objective(problem_context, "linear")

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
    problem_context = ProblemContext(problem=problem)
    D = problem.n_inputs

    N = (
        len(
            get_formula_from_string(
                model_type="linear", problem_context=problem_context
            ).terms
        )
        - n_zero_eigvals(problem_context=problem_context, model_type="linear")
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
    problem_context = ProblemContext(problem=problem)
    D = problem.n_inputs

    N = (
        len(
            get_formula_from_string(
                problem_context=problem_context, model_type="linear"
            ).terms
        )
        + 3
    )
    A = find_local_max_ipopt(problem, "linear")
    assert A.shape == (N, D)


def test_find_local_max_ipopt_results():
    # define problem: no NChooseK constraints
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
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt[:-1]:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])

    # define problem: with NChooseK constraints
    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3"], max_active=1),
        ],
    )

    with pytest.warns(UserWarning):
        A = find_local_max_ipopt(
            problem, "fully-quadratic", ipopt_options={"maxiter": 100}
        )
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])


def test_find_local_max_ipopt_mixed_runs():
    # define problem: no NChooseK constraints
    inputs = [opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]
    inputs.append(opti.Categorical(f"x{4}", ["a", "b"]))
    inputs.append(opti.Discrete(f"x{5}", [1, 2, 3]))
    problem = opti.Problem(
        inputs=opti.Parameters(inputs),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2"], rhs=1),
            opti.LinearInequality(["x2"], lhs=[-1], rhs=-0.1),
            opti.LinearInequality(["x1", "x2"], lhs=[5, 4], rhs=3.9),
            opti.LinearInequality(["x1", "x2"], lhs=[-20, 5], rhs=-3),
        ],
    )

    np.random.seed(1)
    find_local_max_ipopt(problem=problem, model_type="linear", n_experiments=12)


def test_find_local_max_ipopt_sampling():
    # define problem
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
    )

    # test sampling methods
    find_local_max_ipopt(problem, "linear", sampling=OptiSampling)
    find_local_max_ipopt(problem, "linear", sampling=CornerSampling)
    find_local_max_ipopt(problem, "linear", sampling=ProbabilitySimplexSampling)
    sampling = np.zeros(shape=(10, 3)).flatten()
    find_local_max_ipopt(problem, "linear", n_experiments=10, sampling=sampling)


def test_find_local_max_ipopt_discrete():
    # define problem: no NChooseK constraints
    inputs = [opti.Continuous("x1", [0, 1]), opti.Discrete("x2", [0.2, 0.8])]
    problem = opti.Problem(
        inputs=opti.Parameters(inputs),
        outputs=[opti.Continuous("y")],
        constraints=[],
    )

    np.random.seed(1)
    data = find_local_max_ipopt(
        problem=problem, model_type="fully-quadratic"
    ).to_numpy()

    correct_data = np.array(
        [
            [1.0, 0.2],
            [1.0, 0.8],
            [0.0, 0.2],
            [0.0, 0.8],
            [0.5, 0.2],
            [0.5, 0.8],
        ]
    )

    assert np.shape(data) == (9, 2)
    for row in data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in data])


def test_find_local_max_ipopt_categorical():
    # define problem: no NChooseK constraints
    inputs = [opti.Continuous("x1", [0, 1]), opti.Categorical("x2", ["a", "b"])]
    problem = opti.Problem(
        inputs=opti.Parameters(inputs),
        outputs=[opti.Continuous("y")],
        constraints=[],
    )

    np.random.seed(1)
    data = find_local_max_ipopt(
        problem=problem, model_type="fully-quadratic"
    ).to_numpy()

    correct_data = np.array(
        [
            [1.0, "a"],
            [1.0, "b"],
            [0.0, "a"],
            [0.0, "b"],
            [0.5, "a"],
            [0.5, "b"],
        ]
    )

    assert np.shape(data) == (8, 2)
    for row in data:
        assert np.any(
            [np.allclose(row[0], float(_row[0]), atol=2e-3) for _row in correct_data]
        )
        assert np.any([row[1] == _row[1] for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(float(row[0]), _row[0], atol=2e-3) for _row in data])
        assert np.any([row[1] == _row[1] for _row in data])


def test_find_local_max_ipopt_fixed_experiments():
    # define problem: no NChooseK constraints, 1 fixed_experiment
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
    A = find_local_max_ipopt(
        problem,
        "linear",
        n_experiments=12,
        fixed_experiments=[[0.3, 0.5, 0.2]],
    )
    opt = np.array(
        [
            [0.2, 0.2, 0.6],
            [0.3, 0.6, 0.1],
            [0.7, 0.1, 0.2],
            [0.3, 0.1, 0.6],
            [0.3, 0.5, 0.2],
        ]
    )
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt[:-1]:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])
    assert np.allclose(A.to_numpy()[0, :], np.array([0.3, 0.5, 0.2]))

    # define problem: no NChooseK constraints, invalid proposal
    with pytest.raises(ValueError):
        find_local_max_ipopt(
            problem,
            "linear",
            n_experiments=12,
            fixed_experiments=np.ones(shape=(12, 3)),
        )

    # define problem: with NChooseK constraints, 2 fixed_experiments
    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3"], max_active=1),
        ],
    )

    with pytest.warns(UserWarning):
        A = find_local_max_ipopt(
            problem,
            "fully-quadratic",
            ipopt_options={"maxiter": 100},
            fixed_experiments=[[1, 0, 0], [0, 1, 0]],
        )
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])
    assert np.allclose(A.to_numpy()[:2, :], opt[:2, :])


def test_check_fixed_experiments():
    # define problem: everything fine
    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3"], max_active=1),
        ],
    )
    fixed_experiments = np.array([[1, 0, 0], [0, 1, 0]])
    check_fixed_experiments(problem, 3, fixed_experiments)

    # define problem: not enough experiments
    fixed_experiments = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        check_fixed_experiments(problem, 2, fixed_experiments)

    # define problem: invalid shape
    fixed_experiments = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    with pytest.raises(ValueError):
        check_fixed_experiments(problem, 3, fixed_experiments)


def test_check_constraints_and_domain_respected():
    # problem with unfulfillable constraints
    inputs = ["x1", "x2", "x3"]
    # formulation constraint
    constr1 = opti.LinearEquality(names=inputs, rhs=1)
    problem = opti.Problem(
        inputs=[opti.Continuous(x, domain=[0.5, 1]) for x in inputs],
        outputs=[opti.Continuous("y")],
        constraints=[constr1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = find_local_max_ipopt(problem=problem, model_type="linear")

    with pytest.warns(UserWarning, match="Please check your results"):
        check_constraints_and_domain_respected(problem, A, 0)

    # everything ok
    problem = opti.Problem(
        inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2", "x3"], rhs=1),
            opti.NChooseK(names=["x1", "x2", "x3"], max_active=1),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = find_local_max_ipopt(problem=problem, model_type="linear")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_constraints_and_domain_respected(problem, A, 0)
