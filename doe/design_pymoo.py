from copy import deepcopy

import numpy as np
import opti
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem as pymooProblem
from pymoo.optimize import minimize

from doe.design import get_formula_from_string, get_objective, n_ignore_eigvals

# opti problem
ndim = 15
problem = opti.Problem(
    inputs=opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(ndim)]),
    outputs=[opti.Continuous("y")],
    constraints=[opti.LinearEquality(names=[f"x{i+1}" for i in range(ndim)], rhs=1)],
)
model = get_formula_from_string(problem, "fully-quadratic", rhs_only=True)
n_experiments = len(model.terms) + 2  # - n_ignore_eigvals(problem, model_formula)


# define pymoo problem class
class MyProblem(pymooProblem):
    def __init__(self, problem: opti.Problem, model_type, n_experiments, **kwargs):
        self.problem = deepcopy(problem)
        self.model = get_formula_from_string(problem, model_type, rhs_only=True)
        self.objective = get_objective(problem, model)
        self.ndim = problem.n_inputs
        self.n_experiments = n_experiments
        self.n_ignore_eigvals = n_ignore_eigvals(problem, model_type)

        super().__init__(
            n_var=ndim * n_experiments,
            n_obj=1,
            n_constr=n_experiments,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        popsize = x.shape[0]

        x = x.reshape(popsize * self.n_experiments, ndim)

        X = self.model.get_model_matrix(
            pd.DataFrame(x, columns=self.problem.inputs.names)
        ).to_numpy()

        X = X.reshape(popsize, self.n_experiments, len(self.model.terms))

        # compute objective
        F = -np.sum(
            np.log(
                np.linalg.eigvalsh(np.transpose(X, axes=[0, 2, 1]) @ X)[
                    :, self.n_ignore_eigvals :
                ]
            ),
            axis=1,
        )

        # compute constraint violation
        x = x.reshape(popsize, self.n_experiments, self.ndim)
        G = np.abs(np.sum(x, axis=2) - 1) - 1e-2
        G[G <= 0] = 0

        out["F"] = F
        out["G"] = G


_problem = MyProblem(problem, "fully-quadratic", n_experiments)
# _problem = ConstraintsAsPenalty(_problem)

algorithm = GA()

res = minimize(
    _problem,
    algorithm,
    seed=1,
    verbose=True,
    return_least_infeasible=True,
    termination=("n_gen", 3000),
)

print(np.round(res.X.reshape(n_experiments, ndim), 3))
