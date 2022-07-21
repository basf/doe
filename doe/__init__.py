# flake8: noqa

__version__ = "0.1"

from doe import design, jacobian, utils
from doe.design import find_local_max_ipopt
from doe.jacobian import JacobianForLogdet
from doe.utils import get_formula_from_string, n_zero_eigvals
