import numpy as np
import pandas as pd

"""
    Replicate GCTA LMM
"""

# function generating V as variance of all components
def generate_V(grm: np.ndarray, var_i: float, var_p: float, n_indiv: int):
    print("Generating V. \n")
    var_e = var_p - var_i
    V = grm*var_i + np.identity(n_indiv)*var_e
    # check if V is invertible 
    if np.linalg.det(V) == 0:
        raise ValueError("Error. V is not invertible.")
    print("Successfully generating V. \n")
    return V

# function generating P as projection of variance excluding covariates
def generate_P(y:np.ndarray, X:np.ndarray, grm: np.ndarray, var_i: float, var_p: float, n_indiv: int):
    print("Generating P. \n")
    V = generate_V(grm, var_i, var_p, n_indiv)
    V_i = np.linalg.inv(V)
    covar_p = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_i), X))), np.transpose(X))
    P = V_i - np.matmul(np.matmul(V_i, covar_p), V_i)
    print("Successfully generating P. \n")
    return P


def generate_grm(geno: np.ndarray, n_indiv: int, n_snp: int):
    """Genetic Relation Matrix - A 
    Calculate the partial derivative matrix. 

    Parameters
    ----------
    geno : List[np.ndarray]
        Genotype, in shape of [n_indiv, n_snp]
    maf : List[np.ndarray]
        Minor allele frequency, in shape of [n_indiv, n_snp]
    n_indiv : int
        Number of snps for each individua
    n_snp : int
        Number of individua
    Returns
    -------
    grm : List[np.ndarray]
        grm, in shape of [n_indiv, n_snp] 
    """
    print("Generating GRM. \n")
    geno = geno.drop(columns=geno.columns[(geno.mean(axis=0)/2 == 0) | (geno.mean(axis=0)/2 > 0.5)])
    maf = geno.mean(axis=0)/2
    maf = maf.to_numpy()
    geno=(geno-geno.mean())/geno.std()

    A = np.zeros((n_indiv, n_indiv))
    for i in range(n_indiv):
        for j in range(n_indiv):
            if i > j:
                A[i, j] = sum((geno.iloc[i,x] - 2*maf[x])*(geno.iloc[j,x] - 2*maf[x])/(2*maf[x]*(1-maf[x])) for x in range(n_snp)) / n_snp
    A = A.transpose() + A
    np.fill_diagonal(A, 1)
    A[A<0] = 0
    print("Successfully generating GRM. \n")
    return A 

def generate_loglikelihood(y: np.ndarray, X: np.ndarray, P: np.ndarray, V: np.ndarray):
    L = - ()/2
    return