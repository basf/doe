# Introduction

doe is a python package for the computation of (D-)optimal experimental designs. It uses [opti](https://basf.github.io/mopti/) for experiment specification and adding domain knowledge and [formulaic](https://matthewwardrop.github.io/formulaic/).


Opti allows to define an arbitrary number of decision variables using <code>Problem</code> objects. These can take values corresponding to their type and domain, e.g.

* continuous: $x_1 \in [0, 1]$
* discrete: $x_2 \in \{1, 2, 5, 7.5\}$
* categorical: $x_3 \in \{A, B, C\}$.

!!! warning
    Discrete and categorical variables cannot currently be constrained.

Additionally, constraints on the values of the decision variables can be taken into account, e.g.

* linear equality: $\sum x_i = 1$
* linear inequality: $2 x_1 \leq x_2$
* non-linear equality: $\sum x_i^2 = 1$
* non-linear inequality: $\sum x_i^2 \leq 1$
* n-choose-k: only $k$ out of $n$ parameters can take non-zero values.

The model to be fitted can be specified using formulaic's <code>Formula</code> objects, strings following Wilkinson notation or - with the context of the problem specification - using certain keywords like <code>"linear"</code> or <code>"fully-quadratic"</code>.
