# doe

doe is a python package for the computation of (D-)optimal experimental designs. It uses [opti](https://basf.github.io/mopti/) for experiment specification and adding domain knowledge and [formulaic](https://matthewwardrop.github.io/formulaic/).

## Install

pyreto can be installed from [nexus.roqs.basf.net](https://developer.docs.basf.net/setup/python/#configure) (NOT YET).
```
pip install doe
```
Please make sure to have cyipopt installed. The easiest way to get this package is using
```
conda install -c conda-forge cyipopt
```
See [this link](https://cyipopt.readthedocs.io/en/stable/install.html) for more information on other ways to install cyipopt.

## Usage

```python
import opti
import doe

problem = opti.Problem(
   inputs = opti.Parameters([opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)]),
   outputs = [opti.Continuous("y")],
   constraints = [
       opti.LinearEquality(names=["x1","x2","x3"], rhs=1),
       opti.LinearInequality(["x2"], lhs=[-1], rhs=-0.1),
       opti.LinearInequality(["x3"], lhs=[1], rhs=0.6),
       opti.LinearInequality(["x1","x2"], lhs=[5,4], rhs=3.9),
       opti.LinearInequality(["x1","x2"], lhs=[-20,5], rhs=-3)
   ]
)

res = find_local_max_ipopt(problem, "linear")
```
<br>
<img height="500", src="https://gitlab.roqs.basf.net/bayesopt/doe/-/tree/main/docs/assets/getting_started_constraints_local_opt.PNG" />
<br>