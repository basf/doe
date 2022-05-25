# Tentative NChooseK constraint support

!!! warning
    This way of handling NChooseK constraint is not the latest anymore. By default, find_local_max_ipopt is using nchoosek_handling="as_bounds" now. This causes IPOPT to not change decision variables that have been set to 0 in the beginning because of NChooseK constraints.

doe also supports problems with NChooseK constraints. Since IPOPT has problems finding feasible solutions
using the gradient of the NChooseK constraint violation, a closely related linear constraint that suffices
to fulfill the NChooseK constraint is generated and used for the optimization instead: For each experiment $j$
N-K decision variables $x_{i_1,j},...,x_{i_{N-K,j}}$ from the NChooseK constraints' names attribute are picked
that are forced to be zero. Given the design $x$ is expressed as one lone vector, the linear constraint can then be written as
$$
(0,...,0,x_{i_1,1},0,...,0,x_{i_{N-K},n_{exp}},0,...,0) \cdot x \leq 0
$$
However, this linear constraint is stricter than the original NChooseK constraint. In combination with other
constraints on the same decision variables this can result in a situation where the constraints cannot be fulfilled
even though the original constraints would allow for a solution. For example consider a problem with three decision
variables $x_1, x_2, x_3, x_4$, an NChooseK constraint on all three variable that restricts the number of active constraints
to two. Additionally, we have a linear constraint
$$
x_3 + x_4 \geq 0.1
$$
We can easily find points that fulfill both constraints (e.g. $(0,0,0,0.1)$). Now consider the stricter, linear constraint
from above. Eventually, it will happen that $x_3$ and $x_4$ are chosen to be zero for one experiment. For this experiment
it is impossible to fulfill the linear constraint $x_3 + x_4 \geq 0.1$ since $x_3 = x_4 = 0$.

Therefore one has to be very careful when imposing linear constraints upon decision variables that already show up in an NChooseK constraint.

For practical reasons it necessary that the lower bounds of variables affected by NChooseK constraints are zero and the two NChooseK constraints must not share any variables.

You can find an example for a problem with NChooseK constraints and additional linear constraints imposed on the same variables.

```python 
import opti
from doe import find_local_max_ipopt

problem = opti.Problem(
    inputs = [opti.Continuous(f"x{i+1}", [0,1]) for i in range(8)],
    outputs = [opti.Continuous("y")],
    constraints = [
        opti.LinearEquality(names=[f"x{i+1}" for i in range(8)], rhs=1),
        opti.NChooseK(names=["x1","x2","x3"], max_active=1),
        opti.LinearInequality(names=["x1","x2","x3"], rhs=0.7),
        opti.LinearInequality(names=["x7","x8"], lhs=-1, rhs=-0.1),
        opti.LinearInequality(names=["x7","x8"], rhs=0.9),
    ]
)

res = find_local_max_ipopt(
    problem=problem,
    model_type="fully-quadratic",
    ipopt_options={"maxiter":500, "disp":5},
    nchoosek_handling="as_linear_constraint",
)
```

Running these lines of codes yields the following design.

|x1  |x2  |x3  |x4  |x5  |x6  |x7  |x8  |
|----|----|----|----|----|----|----|----|
|0.00|0.00|0.00|0.90|0.00|0.00|0.10|0.00|
|0.36|0.00|0.00|0.00|0.00|0.00|0.00|0.64|
|0.00|0.00|0.00|0.00|0.90|0.00|0.00|0.10|
|0.00|0.00|0.00|0.00|0.45|0.45|0.10|0.00|
|0.45|0.00|0.00|0.45|0.00|0.00|0.10|0.00|
|0.00|0.00|0.00|0.90|0.00|0.00|0.00|0.10|
|0.00|0.45|0.00|0.45|0.00|0.00|0.00|0.10|
|0.70|0.00|0.00|0.00|0.00|0.00|0.00|0.30|
|0.00|0.00|0.00|0.00|0.00|0.48|0.00|0.52|
|0.00|0.00|0.00|0.45|0.00|0.45|0.08|0.02|
|0.69|0.01|0.00|0.00|0.00|0.20|0.10|0.00|
|0.00|0.00|0.46|0.00|0.00|0.00|0.54|0.00|
|0.00|0.00|0.00|0.00|0.90|0.00|0.10|0.00|
|0.51|0.00|0.00|0.00|0.00|0.00|0.50|0.00|
|0.00|0.00|0.39|0.00|0.00|0.00|0.00|0.62|
|0.00|0.36|0.00|0.00|0.00|0.55|0.00|0.10|
|0.10|0.00|0.00|0.00|0.00|0.00|0.90|0.00|
|0.00|0.00|0.70|0.00|0.00|0.00|0.00|0.30|
|0.00|0.54|0.00|0.00|0.00|0.00|0.00|0.46|
|0.00|0.00|0.00|0.00|0.53|0.00|0.00|0.48|
|0.00|0.00|0.00|0.00|0.10|0.00|0.00|0.90|
|0.00|0.00|0.00|0.49|0.00|0.00|0.51|0.00|
|0.00|0.00|0.00|0.00|0.10|0.00|0.90|0.00|
|0.00|0.00|0.45|0.45|0.00|0.00|0.04|0.06|
|0.00|0.00|0.00|0.45|0.45|0.00|0.03|0.07|
|0.45|0.00|0.00|0.00|0.45|0.00|0.10|0.00|
|0.00|0.00|0.70|0.00|0.00|0.20|0.10|0.00|
|0.00|0.00|0.00|0.00|0.00|0.09|0.45|0.45|
|0.44|0.00|0.00|0.00|0.00|0.46|0.00|0.10|
|0.00|0.00|0.45|0.00|0.45|0.00|0.10|0.00|
|0.00|0.43|0.00|0.00|0.00|0.00|0.57|0.00|
|0.00|0.00|0.00|0.49|0.00|0.00|0.00|0.51|
|0.00|0.00|0.00|0.00|0.00|0.49|0.51|0.00|
|0.00|0.70|0.00|0.00|0.00|0.20|0.10|0.00|
|0.00|0.00|0.00|0.00|0.00|0.90|0.00|0.10|
|0.00|0.00|0.00|0.00|0.00|0.90|0.10|0.00|
|0.00|0.45|0.00|0.00|0.45|0.00|0.00|0.10|
|0.00|0.00|0.00|0.00|0.50|0.00|0.50|0.00|
|0.00|0.00|0.40|0.00|0.00|0.50|0.00|0.10|
