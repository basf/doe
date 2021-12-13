from doe.dopt import num_experiments
import opti


def test_num_experiments_continuous():
    # 5 continous inputs
    prob = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(5)], outputs=[opti.Continuous("y")]
    )

    assert num_experiments(prob, "linear") == 6
    assert num_experiments(prob, "linear-and-quadratic") == 11
    assert num_experiments(prob, "linear-and-interactions") == 16
    assert num_experiments(prob, "fully-quadratic") == 21

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

    assert num_experiments(prob, "linear") == 5
    assert num_experiments(prob, "linear-and-quadratic") == 9
    assert num_experiments(prob, "linear-and-interactions") == 11
    assert num_experiments(prob, "fully-quadratic") == 15

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

    assert num_experiments(prob, "linear") == 7
    assert num_experiments(prob, "linear-and-quadratic") == 9
    assert num_experiments(prob, "linear-and-interactions") == 16
    assert num_experiments(prob, "fully-quadratic") == 18

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

    assert num_experiments(prob, "linear") == 8
    assert num_experiments(prob, "linear-and-quadratic") == 11
    assert num_experiments(prob, "linear-and-interactions") == 23
    assert num_experiments(prob, "fully-quadratic") == 26

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

    assert num_experiments(prob, "linear") == 8
    assert num_experiments(prob, "linear-and-interactions") == 25

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

    assert num_experiments(prob, "linear") == 10
    assert num_experiments(prob, "linear-and-quadratic") == 13
    assert num_experiments(prob, "linear-and-interactions") == 42
