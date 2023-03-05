import numpy as np
import opti
import pytest

from doe.sampling import (
    CornerSampling,
    DomainUniformSampling,
    OptiSampling,
    ProbabilitySimplexSampling,
    Sampling,
)


def test_Sampling():
    # define problem
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1])], outputs=[opti.Continuous("y")]
    )
    s = Sampling(problem)
    with pytest.raises(NotImplementedError):
        s.sample(1)


def test_OptiSampling():
    # define problem
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1]), opti.Continuous("x2", [0, 1])],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearInequality(["x1", "x2"], rhs=0.5)],
    )
    s = OptiSampling(problem)

    x = s.sample(10)
    assert np.shape(x) == (20,)
    x = x.reshape((10, 2))
    assert all([np.sum(row) <= 0.51 for row in x])


def test_CornerSampling():
    # define problem
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1]), opti.Continuous("x2", [0, 1])],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearInequality(["x1", "x2"], rhs=0.5)],
    )
    s = CornerSampling(problem)

    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    x = s.sample(4).reshape((4, 2))
    assert all([corner in x for corner in corners])
    assert all([corner in corners for corner in x])


def test_ProbabilitySimplexSampling():
    # define problem
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 2]),
            opti.Continuous("x3", [0, 3]),
        ],
        outputs=[opti.Continuous("y")],
    )
    s = ProbabilitySimplexSampling(problem)

    # check corners come first
    corners = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    x = s.sample(3).reshape((3, 3))
    assert all([corner in x for corner in corners])
    assert all([corner in corners for corner in x])

    # check points are on simplex
    corners = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    x = s.sample(10).reshape((10, 3))
    assert np.allclose(np.sum(x / np.array([1, 2, 3]), axis=1), 1)

    # define problem: invalid bounds
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 2]),
            opti.Continuous("x3", [1, 3]),
        ],
        outputs=[opti.Continuous("y")],
    )
    with pytest.raises(ValueError):
        s = ProbabilitySimplexSampling(problem)


def test_DomainUniformSampling():
    # define problem
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 2]),
            opti.Continuous("x3", [1, 3]),
        ],
        outputs=[opti.Continuous("y")],
    )
    s = DomainUniformSampling(problem)
    x0 = s.sample(50)

    # check sample size
    assert len(x0) == 150

    # check domain is respected
    x0 = x0.reshape(50, 3)
    assert np.all(x0[:, 0] >= 0) and np.all(x0[:, 0] <= 1)
    assert np.all(x0[:, 1] >= 0) and np.all(x0[:, 1] <= 2)
    assert np.all(x0[:, 2] >= 1) and np.all(x0[:, 2] <= 3)
