import pandas as pd
import numpy as np
from ._model import *

def generate_AI(y:np.ndarray, X: np.ndarray, var_i: float, grm: np.ndarray, n_indiv: int):
    """Average Information matrix for one iteration
    Update theta estimate of variance by one iteration of AI alg. 

    Parameters
    ----------
    y : List[np.ndarray]
        Observation, in shape of [n_indiv, 1]
    X : List[np.ndarray]
        Covariates matrix, in shape of [n_indiv, n_covar]
    var_i : float
        Variace of genetic component
    grm : List[np.ndarray]
       Genetic relationship matrix - A, in shape of [n_indiv, n_indiv]
    n_indiv : int
        Number of individuals

    Returns
    -------
    m : List[np.ndarray]
        AI matrix, 2*2 
    """
    var_p = np.var(y)
    P = generate_P(y, X, grm, var_i, var_p, n_indiv)
    m_11 = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(y), P), grm), P), grm), P), y)
    m_12 = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(y), P), grm), P), P), y)
    m_21 = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(y), P), P), grm), P), y)
    m_22 = np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(y), P), P), P), y)
    m = np.array([m_11, m_12, m_21, m_22])
    m = m.reshape(2,2)
    return m


def generate_partial(y:np.ndarray, X: np.ndarray, var_i: float, grm: np.ndarray, n_indiv: int):
    """Partial derivatives of L respect to theta 
    Calculate the partial derivative matrix. 

    Parameters
    ----------
    y : List[np.ndarray]
        Observation, in shape of [n_indiv, 1]
    X : List[np.ndarray]
        Covariates matrix, in shape of [n_indiv, n_covar]
    var_i : float
        Variace of genetic component
    grm : List[np.ndarray]
       Genetic relationship matrix - A, in shape of [n_indiv, n_indiv]
    n_indiv : int
        Number of individuals

    Returns
    -------
    delta : List[np.ndarray]
        partial derivative matrix, 2*1 
    """
    var_p = np.var(y)
    P = generate_P(y, X, grm, var_i, var_p, n_indiv)
    d_1 = np.trace(np.matmul(P, grm)) - np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(y), P), grm), P), y)
    d_2 = np.trace(P) - np.matmul(np.matmul(np.matmul(np.transpose(y), P), P), y)
    d = np.array([d_1, d_2])
    delta = -np.transpose(d)/2
    return delta



