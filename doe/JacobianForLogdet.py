from formulaic import Formula
import opti
from copy import deepcopy
from typing import Callable, Optional, List
import numpy as np
import pandas as pd
import scipy as sp
import time

#TODO: testen
class JacobianForLogdet:
    """A class representing the jacobian/gradient of logdet(X.T@X) w.r.t. the inputs.
    It can be divided into two parts, one for logdet(X.T@X) w.r.t. X (there is a simple
    closed expression for this one) and one model dependent part for the jacobian of X.T@X
    w.r.t. the inputs. Because each row of X only depends on the inputs of one experiment
    the second part can be formulated in a simplified way. It is built up with n_experiment
    blocks of the same structure which is represended by the attribute jacobian_building_block.
    """

    def __init__(
        self,
        problem: opti.Problem,
        model: Formula,
        n_experiments: int,
        jacobian_building_block: Optional[Callable] = None,
        delta: Optional[float] = 1e-3,
    ) -> None:
        """
        Args:
            problem (opti.Problem): An opti problem defining the DoE problem together with model_type.
            model_type (str or Formula): A formula containing all model terms.
            n_experiments (int): Number of experiments
            jacobian_building_block (Callable): A function that returns the jacobian building block for 
                the given model. By default the default_jacobian_building_block is chosen. For models of
                higher than quadratic order one needs to provide an own jacobian.
            delta (float): A regularization parameter for the information matrix. Default value is 1e-3.

        """

        self.model = deepcopy(model)
        self.problem = deepcopy(problem)
        self.n_experiments = n_experiments
        self.delta = delta

        self.vars = self.problem.inputs.names
        self.n_vars = self.problem.n_inputs

        self.model_terms = np.array(model.terms, dtype=str)
        self.n_model_terms = len(self.model_terms)

        if jacobian_building_block is not None:
            self.jacobian_building_block = jacobian_building_block
        else:
            self.jacobian_building_block = default_jacobian_building_block(self.vars, self.model_terms)

    def jacobian(self, x:np.ndarray) -> np.ndarray:
        t = time.time()
        """Computes the full jacobian for the given input."""

        #get model matrix X
        x = x.reshape(self.n_experiments, self.n_vars)
        X = pd.DataFrame(x, columns=self.vars)
        X = self.model.get_model_matrix(X).to_numpy()

        #first part of jacobian
        J1 = -2 * X @ sp.linalg.solve(X.T @ X + self.delta* np.eye(self.n_model_terms), np.eye(self.n_model_terms), sym_pos=True, overwrite_b=True)
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(self.n_experiments, self.n_vars, self.n_model_terms)

        #second part of jacobian
        J2 = np.empty(shape=(self.n_experiments, self.n_vars, self.n_model_terms))
        for i in range(self.n_experiments):
            J2[i,:,:] = self.jacobian_building_block(x[i,:])
        
        #combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()

#TODO: testen
def default_jacobian_building_block(vars: List[str], model_terms: List[str]) -> Callable:
    """Returns a function that returns the terms of the reduced jacobian for one experiment.

    Args:
        vars (List[str]): List of variable names of the model
        model_terms (List[str]): List of model terms saved as string.

    Returns:
        A function that returns a jacobian building block usable for models up to second order. 
    """

    n_vars = len(vars)

    #find the term names
    terms = ["1"]
    for name in vars:
        terms.append(name)
    for name in vars:
        terms.append(name+"**2")
    for i in range(n_vars):
        for j in range(i+1,n_vars):
            term = str(Formula(vars[j]+":"+vars[i]+"-1").terms[0])
            terms.append(term)

    def jacobian_building_block(x: np.ndarray) -> np.ndarray:
        """Computes the jacobian building block for a single experiment with inputs x."""
        B = np.zeros(shape=(n_vars, len(terms)))

        #derivatives of intercept term are zero

        #derivatives of linear terms
        B[:,1:n_vars+1] = np.eye(n_vars)

        #derivatives of quadratic terms
        B[:,n_vars+1:2*n_vars+1] = 2*np.diag(x)

        #derivatives of interaction terms
        col = 2*n_vars+1
        for (i,name) in enumerate(vars[:-1]):
            n_terms = len(vars[i+1:])
            B[i,col:col+n_terms] = x[i+1:]
            B[i+1:,col:col+n_terms] = x[i] * np.eye(n_terms)
            col += n_terms

        return pd.DataFrame(B, columns=terms)[model_terms].to_numpy()

    return jacobian_building_block
