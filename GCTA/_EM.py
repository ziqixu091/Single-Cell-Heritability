import pandas as pd
import numpy as np
from _model import *

def EM(y:np.ndarray, X:np.ndarray, grm: np.ndarray, var_i: float, var_p: float, n_snp: int, n_indiv: int):
    """Expectation Maximization for one iteration
    Update theta estimate of variance by one iteration of EM alg, with an arbitrary initial value. 

    Parameters
    ----------
    y : List[np.ndarray]
        Observation, in shape of [n_indiv, 1]
    X : List[np.ndarray]
        Covariates matrix, in shape of [n_indiv, n_covar]
    grm : List[np.ndarray]
       Genetic relationship matrix - A, in shape of [n_indiv, n_indiv]
    var_i : float
        Variance of genetic component
    var_p : float
        Variance of observations
    n_snp : int
        Number of snps for each individual 
    n_indiv : int
        Number of individuals

    Returns
    -------
    var_i_new: float
        Updated estimation of variances 

    """

    print("Processing EM iteration. \n")
    P = generate_P(y, X, grm, var_i, var_p, n_indiv)
    papy = np.matmul(np.matmul(np.matmul(P, grm), P), y)
    tr_term = pow(var_i, 2)*np.identity(n_indiv) - pow(var_i, 4)*np.matmul(P, grm)

    var_i_new =  (pow(var_i, 4)*np.matmul(np.transpose(y), papy) + np.trace(tr_term)) / n_snp
    print("Successfully generating EM estimates. \n")
    return var_i_new




