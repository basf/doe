# DoE

DoE is a python package for generating (D-)optimal experimental designs. 
It uses [opti](https://basf.github.io/mopti/) and [formulaic](https://matthewwardrop.github.io/formulaic/) for specifying the design space and model.

You can find the documentation [here](https://basf.github.io/doe).


## Install

DoE can be installed with 
```
pip install git+https://github.com/basf/doe.git
```

Please make sure to have cyipopt installed.
With conda the easiest way to get this package is using
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

design = find_local_max_ipopt(problem, "linear")
```

![doe_example](docs/assets/getting_started_constraints_local_opt.PNG)

---

### Code contributors
<a href="https://github.com/basf/doe/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=basf/doe" />
</a>

and of course https://github.com/Osburg!

### Math contributors
* David Hajnal
* Jorge Diaz