import numpy as np
import pandas as pd

"""
    Replicate GCTA LMM
"""

# function generating V as variance of all components
def generate_V(grm: np.ndarray, var_i: float, var_p: float, n_indiv: int):
    var_e = var_p - var_i
    V = grm*var_i + np.identity(n_indiv)*var_e
    # check if V is invertible 
    if np.linalg.det(V) == 0:
        raise ValueError("Error. V is not invertible.")
    return V

# function generating P as projection of variance excluding covariates
def generate_P(y:np.ndarray, X:np.ndarray, grm: np.ndarray, var_i: float, var_p: float, n_indiv: int):
    V = generate_V(grm, var_i, var_p, n_indiv)
    V_i = np.linalg.inv(V)
    covar_p = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_i), X))), np.transpose(X))
    P = V_i - np.matmul(np.matmul(V_i, covar_p), V_i)
    return P


def generate_grm(geno: np.ndarray, maf:np.ndarray):
    """Genetic Relation Matrix - A 
    Calculate the partial derivative matrix. 

    Parameters
    ----------
    geno : List[np.ndarray]
        Genotype, in shape of [n_indiv, n_snp]
    maf : List[np.ndarray]
        Minor allele frequency, in shape of [n_indiv, n_snp]

    Returns
    -------
    grm : List[np.ndarray]
        grm, in shape of [n_indiv, n_snp] 
    """
    return 