# Multiple mixture constraints

In this example we want to obtain a design with 125 experiments for the following problem with two mixture constraints

```python
problem = opti.Problem(
    inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(12)]
    + [opti.Continuous(f"p{i+1}", [-1, 1]) for i in range(3)],
    outputs=[opti.Continuous("y")],
    constraints=[
        opti.LinearEquality(names=[f"x{i+1}" for i in range(6)], rhs=1),
        opti.LinearEquality(names=[f"x{i+1}" for i in range(6, 12)], rhs=1),
    ],
)

res = find_local_max_ipopt(
    problem=problem,
    model_type="fully-quadratic",
    n_experiments=125,
    ipopt_options={"maxiter":2000},
)
```

We obtain the following design.

|x1  |x2  |x3  |x4  |x5  |x6  |x7  |x8  |x9  |x10 |x11 |x12 |p1   |p2   |p3   |
|----|----|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|
|0.56|0.44|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 0.29|-1.00|-1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|-0.47|-1.00|-1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.42|0.59|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|1.00|0.00|0.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.00|0.38|0.00|0.62|0.00|0.51|0.00|0.00|0.00|0.50|0.00|-1.00| 0.12| 0.23|
|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 0.09|-1.00|-1.00|
|0.00|0.00|0.19|0.81|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|-1.00|-1.00|-1.00|
|0.51|0.00|0.50|0.00|0.00|0.00|0.00|0.48|0.00|0.52|0.00|0.00|-1.00| 0.08|-1.00|
|0.00|0.00|0.47|0.53|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.50|0.00|0.00|0.00|0.50|0.00|0.52|0.00|0.00|0.00|0.48|-1.00| 1.00|-1.00|
|0.00|0.89|0.00|0.10|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.58|0.00|0.42|0.00|0.00|1.00|0.00|0.00|0.00| 1.00|-1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|-1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.56|0.00|0.44|0.00|0.00|0.51|0.00|0.00|0.49|-1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.41|0.00|0.59|0.00|0.00|0.00|0.00|0.00|0.58|0.42|0.00|0.00| 1.00|-1.00|-1.00|
|0.74|0.00|0.00|0.00|0.00|0.26|0.00|0.73|0.00|0.00|0.27|0.00| 0.18|-1.00|-0.92|
|0.61|0.00|0.39|0.00|0.00|0.00|0.00|0.55|0.45|0.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.49|0.00|0.00|0.51|0.00|0.00|0.00|0.00|0.47|0.00|0.54|-1.00|-1.00| 1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.50|0.00|0.50|0.00|0.00|0.00|0.00|0.38|0.61|0.00| 0.05| 1.00| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|-1.00| 1.00|-0.61|
|0.48|0.00|0.00|0.00|0.53|0.00|0.00|1.00|0.00|0.00|0.00|0.00|-1.00|-1.00|-1.00|
|0.54|0.00|0.00|0.00|0.00|0.46|0.44|0.00|0.00|0.00|0.00|0.56|-1.00|-1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.69|0.30|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.00|0.00|0.44|0.00|0.00|0.56|0.00|0.00|0.43|0.00|0.00|0.57| 1.00| 1.00| 0.25|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|1.00|0.00|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.57|0.43|0.00|0.00|0.51|0.00|0.00|0.00|0.49|0.00|-0.27| 1.00| 1.00|
|0.59|0.00|0.00|0.41|0.00|0.00|0.47|0.00|0.00|0.00|0.00|0.53| 0.01| 1.00| 1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.16|0.00|0.84|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.00|0.53|0.00|0.00|0.47|0.00|0.00|0.39|0.00|0.61|0.00|0.00| 1.00| 1.00|-0.49|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.00|0.78|0.22|0.45|0.00|0.00|0.55|0.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.60|0.00|0.41|0.00|1.00|0.00|0.00|0.00|0.00|-1.00| 1.00|-0.24|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00| 1.00|-1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.59|0.00|0.41|0.00| 1.00|-1.00| 1.00|
|0.00|0.44|0.00|0.56|0.00|0.00|0.00|0.00|0.00|0.53|0.00|0.48| 1.00| 0.06|-1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.54|0.00|0.00|0.00|0.00|0.47|0.00|0.00|0.00|0.48|0.52|0.00|-1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|-1.00|-1.00| 1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00|-1.00|-1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00|-1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00| 1.00|-1.00|
|0.00|0.55|0.00|0.00|0.00|0.45|0.00|0.00|0.00|0.00|0.55|0.45|-0.19| 1.00| 0.24|
|0.76|0.00|0.00|0.23|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00|-0.15| 1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.47|0.54|0.00|0.00|0.00|0.51|0.00|0.49|0.00|0.00|0.00| 0.39|-1.00|-1.00|
|0.00|0.00|0.00|0.52|0.48|0.00|0.54|0.46|0.00|0.00|0.00|0.00| 0.15| 1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00|-1.00| 0.04|
|0.00|0.64|0.00|0.00|0.00|0.36|0.44|0.00|0.00|0.00|0.00|0.56| 1.00|-1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00| 0.02|-1.00|
|0.00|0.00|0.00|0.00|0.53|0.48|0.00|0.50|0.00|0.00|0.00|0.51| 1.00|-1.00| 1.00|
|0.00|0.46|0.00|0.54|0.00|0.00|0.00|0.00|0.00|0.48|0.52|0.00| 1.00|-1.00| 0.50|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.43|0.57|0.00|0.00|0.00|0.57|0.00|0.00|0.00|0.00|0.43|-1.00|-1.00| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00| 0.84|-1.00|-0.83|
|0.45|0.00|0.00|0.00|0.55|0.00|0.48|0.00|0.00|0.53|0.00|0.00|-0.04|-1.00| 1.00|
|0.00|0.54|0.00|0.00|0.46|0.00|0.00|0.58|0.00|0.00|0.42|0.00| 1.00|-0.12| 1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.33|0.66|0.00|0.00|0.00|0.00|0.00|0.49|0.00|0.51|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.56|0.00|0.00|0.00|0.44|0.00|0.00|0.00|1.00|0.00|0.00| 1.00|-0.27| 1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00|-0.09| 1.00|
|0.00|0.00|0.47|0.53|0.00|0.00|0.00|0.50|0.00|0.00|0.00|0.50| 1.00| 0.64| 0.24|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00|-1.00| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.65|0.00|0.35|0.00|0.56|0.00|0.00|0.45|0.00|0.00| 1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|1.00|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|1.00|0.00|0.00|0.00|0.00|0.00| 1.00|-1.00|-1.00|
|0.00|0.00|0.45|0.00|0.55|0.00|0.51|0.00|0.49|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|1.00|0.00|0.00|0.00|0.00| 1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.51|0.00|0.49|0.52|0.49|0.00|0.00|0.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00|-1.00| 1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|-1.00| 1.00| 1.00|
|0.34|0.00|0.00|0.67|0.00|0.00|0.00|0.00|0.54|0.46|0.00|0.00|-1.00|-1.00|-0.01|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.48|0.52|-1.00|-1.00| 1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00|-0.35| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.73|0.00|0.00|0.27|0.00|0.00|1.00|0.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00|-1.00| 1.00|
|0.00|0.53|0.47|0.00|0.00|0.00|0.00|0.00|0.00|0.49|0.00|0.51| 0.57| 1.00|-1.00|
|0.00|0.61|0.00|0.39|0.00|0.00|0.00|0.00|0.64|0.00|0.36|0.00|-1.00|-1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|-1.00|-0.50|-0.70|
|0.00|0.00|0.00|0.53|0.46|0.00|0.00|0.00|0.00|0.00|0.49|0.51| 1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|-1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|-1.00| 1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.49|0.00|0.50|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00|-1.00|-1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00| 1.00| 0.04|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.58|0.43|0.00|0.00|0.00|-1.00| 0.62|-1.00|
|0.41|0.59|0.00|0.00|0.00|0.00|0.48|0.00|0.00|0.52|0.00|0.00| 1.00| 1.00| 0.37|
|0.00|0.00|0.00|0.00|0.45|0.55|0.00|0.00|0.72|0.00|0.28|0.00|-0.32| 0.43| 1.00|
|0.00|0.46|0.00|0.54|0.00|0.00|0.00|0.48|0.52|0.00|0.00|0.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.00|0.69|0.30|0.00|0.00|0.00|0.47|0.53|0.00| 1.00| 1.00| 1.00|
|0.77|0.00|0.00|0.23|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00| 1.00|-1.00|
|0.47|0.53|0.00|0.00|0.00|0.00|0.00|0.00|0.53|0.00|0.00|0.47| 1.00| 1.00|-1.00|
|0.00|0.00|0.00|0.00|0.56|0.44|0.00|0.49|0.00|0.00|0.50|0.00| 1.00| 1.00|-1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00| 1.00| 1.00|-1.00|
|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00| 1.00| 1.00| 1.00|
|0.00|0.00|0.00|0.55|0.44|0.00|0.00|1.00|0.00|0.00|0.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.00|0.00|0.00|1.00|1.00|0.00|0.00|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.40|0.60|0.00|0.00|-1.00| 1.00| 1.00|
|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|0.00|-1.00|-0.86| 1.00|
|1.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00| 1.00| 1.00| 0.83|
|0.49|0.00|0.00|0.51|0.00|0.00|0.46|0.54|0.00|0.00|0.00|0.00|-1.00| 1.00| 1.00|
|0.50|0.00|0.00|0.00|0.50|0.00|0.00|0.00|0.00|0.46|0.00|0.54|-1.00| 1.00| 0.04|
|0.00|0.00|0.49|0.00|0.00|0.51|0.00|0.00|0.54|0.00|0.46|0.00| 1.00| 1.00|-1.00|
|0.00|0.00|0.00|1.00|0.00|0.00|0.00|0.00|0.00|0.00|1.00|0.00|-1.00|-1.00| 1.00|
|0.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|1.00|0.00|0.00|0.00|-1.00|-1.00|-1.00|