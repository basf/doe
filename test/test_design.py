import opti

from doe.design import num_experiments, optimal_design


def test_num_experiments_continuous():
    # 5 continous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)],
        outputs=[opti.Continuous("y")],
    )

    assert num_experiments(problem, "linear", 0) == 6
    assert num_experiments(problem, "linear-and-quadratic", 0) == 11
    assert num_experiments(problem, "linear-and-interactions", 0) == 16
    assert num_experiments(problem, "fully-quadratic", 0) == 21


def test_num_experiments_constrained():
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

    assert num_experiments(prob, "linear", 0) == 5
    assert num_experiments(prob, "linear-and-quadratic", 0) == 9
    assert num_experiments(prob, "linear-and-interactions", 0) == 11
    assert num_experiments(prob, "fully-quadratic", 0) == 15


def test_num_experiments_categorical():
    # 2 continuous inputs, 1 categorical
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Categorical("x3", ["L1", "L2", "L3", "L4", "L5"]),
        ],
        outputs=[opti.Continuous("y")],
    )

    assert num_experiments(prob, "linear", 0) == 7
    assert num_experiments(prob, "linear-and-quadratic", 0) == 9
    assert num_experiments(prob, "linear-and-interactions", 0) == 16
    assert num_experiments(prob, "fully-quadratic", 0) == 18

    # 3 continuous inputs, 1 categorical
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Continuous("x3"),
            opti.Categorical("x4", ["L1", "L2", "L3", "L4", "L5"]),
        ],
        outputs=[opti.Continuous("y")],
    )

    assert num_experiments(prob, "linear", 0) == 8
    assert num_experiments(prob, "linear-and-quadratic", 0) == 11
    assert num_experiments(prob, "linear-and-interactions", 0) == 23
    assert num_experiments(prob, "fully-quadratic", 0) == 26

    # 2 continuous inputs, 2 categoricals
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Categorical("x5", ["L1", "L2", "L3"]),
            opti.Categorical("x6", ["L1", "L2", "L3", "L4"]),
        ],
        outputs=[opti.Continuous("y")],
    )

    assert num_experiments(prob, "linear", 0) == 8
    assert num_experiments(prob, "linear-and-interactions", 0) == 25

    # 3 continuous inputs, 3 categoricals
    prob = opti.Problem(
        inputs=[
            opti.Continuous("x1"),
            opti.Continuous("x2"),
            opti.Continuous("x3"),
            opti.Categorical("x4", ["L1", "L2"]),
            opti.Categorical("x5", ["L1", "L2", "L3"]),
            opti.Categorical("x6", ["L1", "L2", "L3", "L4"]),
        ],
        outputs=[opti.Continuous("y")],
    )

    assert num_experiments(prob, "linear", 0) == 10
    assert num_experiments(prob, "linear-and-quadratic", 0) == 13
    assert num_experiments(prob, "linear-and-interactions", 0) == 42


def test_optimal_design():
    inputs = opti.Parameters([opti.Continuous(f"x{i}", [0, 1]) for i in range(4)])
    problem = opti.Problem(
        inputs=inputs,
        outputs=[opti.Continuous("y")],
        constraints=[opti.NChooseK(inputs.names, max_active=3)],
    )
    D = problem.n_inputs
    N = num_experiments(problem)
    A = optimal_design(problem)
    assert A.shape == (N, D)
