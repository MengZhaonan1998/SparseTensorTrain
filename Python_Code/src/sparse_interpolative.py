import numpy as np
import random as rd
from scipy.sparse import csc_matrix, random
from scipy.linalg import lu
from typing import Tuple, Union, List
from interpolative_decomposition import prrldu, prrldu2
from sparse_opt import csc_col_select, csc_row_select, permutation_vector_to_matrix

def simu_spPrrldu(M: csc_matrix, cutoff: float = 0.0, 
              maxdim: int = np.iinfo(np.int32).max
              ) -> Tuple[csc_matrix, csc_matrix, csc_matrix,
                         csc_matrix, csc_matrix, float]:
    """
    Simulator of partial rank revealing LDU decomposition
    Simulating the sparse operators in LU as there are no
    LU methods handling NON-SQUARE matrices in Scipy/...
    """
    assert maxdim > 0, "maxdim must be positive"
    # splu simulation
    dense_M = M.toarray()
    L, D, U, row_perm_inv, col_perm_inv, inf_error = prrldu(dense_M, cutoff, maxdim)
    L_csc = csc_matrix(L)
    U_csc = csc_matrix(U)
    D_csc = csc_matrix(np.diag(D))
    Pr = permutation_vector_to_matrix(row_perm_inv)
    Pc = permutation_vector_to_matrix(col_perm_inv)
    Pr_csc = csc_matrix(Pr)
    Pc_csc = csc_matrix(Pc)
    return L_csc, D_csc, U_csc, Pr_csc, Pc_csc, inf_error

def simu_spPrrldu2(M: csc_matrix, cutoff: float = 0.0, 
              maxdim: int = np.iinfo(np.int32).max
              ) -> Tuple[csc_matrix, csc_matrix, csc_matrix]:
    """
    Simulator of partial rank revealing LDU decomposition
    Simulating the sparse operators in LU as there are no
    LU methods handling NON-SQUARE matrices in Scipy/...
    """
    assert maxdim > 0, "maxdim must be positive"
    
    # splu simulation
    dense_M = M.toarray()
    P, L, U = lu(dense_M)
    #print(f"bk1-max error: {np.max(np.abs(P @ L @ U - dense_M))}")
    P = csc_matrix(P)
    L = csc_matrix(L)
    U = csc_matrix(U)
    
    # Rank truncation
    d = np.abs(U.diagonal())
    r = len(d[d > cutoff])
    r = r if r <= maxdim else maxdim 
     
    # New L, U (by rank truncation)
    L_new = csc_col_select(L, r)  #L = L[:, 0:r]
    U_new = csc_row_select(U, r)  #U = U[0:r, :]    
    #print(f"bk2-max error: {np.max(np.abs(P.toarray() @ L_new.toarray() @ U_new.toarray() - dense_M))}")
    return P, L_new, U_new

# ! Problem of prrldu2 for sparse matrices !
def unit_test_1():
    rd.seed(10)
    m = 20
    r = 10
    n = 15
    A = random(m, r, density=0.4, format='csc', random_state=15)
    B = random(r, n, density=0.2, format='csc', random_state=30)
    M = A.dot(B)
    M_dense = M.toarray()
    
    # Dense prrldu
    L, D, U, row_perm_inv, col_perm_inv, inf_error = prrldu(M_dense, 1e-10, min(m,n))
    recon = L @ np.diag(D) @ U
    Pr = permutation_vector_to_matrix(row_perm_inv)
    Pc = permutation_vector_to_matrix(col_perm_inv)
    max_err = np.max(np.abs(recon - Pr.T @ M_dense @ Pc))   
    print(f"max error prrldu (dense) = {max_err}")
    
    # Dense prrldu2 (without column pivots)
    P, L, U = prrldu2(M_dense, 1e-10, min(m,n))
    Recon = P @ L @ U
    max_err = np.max(np.abs(Recon - M_dense))
    print(f"max error prrldu2 (dense) = {max_err}")

    # Sparse prrldu (simu)
    L_csc, D_csc, U_csc, Pr_csc, Pc_csc, inf_error = simu_spPrrldu(M, 1e-10, min(m,n))
    recon = L_csc.dot(D_csc.dot(U_csc))
    recon_recover_rc = Pr_csc.dot(recon.dot(Pc_csc.transpose()))
    max_err = np.max(np.abs(recon_recover_rc.toarray() - M_dense))
    print(f"max error prrldu (sparse) = {max_err}")
    return 

unit_test_1()