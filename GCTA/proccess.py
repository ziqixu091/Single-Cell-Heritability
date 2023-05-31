from ._AI import *
from ._EM import *
from ._model import *
import fire 

def gcta_process(y: np.ndarray, X: np.ndarray, geno: np.ndarray, X: np.ndarray, n_snp: int, n_indiv: int):
    
    var_p = y.var()
    var_i = var_p / 2
    n_indiv = y.shape()[0]
    n_snp = y.shape()[1]
    grm = generate_grm(geno, n_indiv, n_snp)
    
    var_i_one = EM(y, X, grm, var_i, var_p, n_snp, n_indiv)

    while(var_i_one):

    return

if __name__ == "__main__":
    fire.Fire(gcta_process)

