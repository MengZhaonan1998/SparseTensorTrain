import numpy as np
import random as rd
from scipy.sparse import csc_matrix, random
from scipy.sparse.linalg import SuperLU, spsolve_triangular
from scipy.linalg import lu
from typing import Tuple, Union, List
from interpolative_decomposition import prrldu, prrldu2
from sparse_opt import csc_col_select, csc_row_select, permutation_vector_to_matrix

def rrSuperLU(M: csc_matrix, cutoff: float = 0.0, 
              maxdim: int = np.iinfo(np.int32).max):
    """
    Rank revealing SuperLU for Sparse LU decomposition
    Args:
        M: Input csc sparse matrix 
        cutoff: truncation threshold
        maxdim: maximum dimension of factors
    Returns:
        Tuple containing (L_csc, U_csc, Pr_csc, Pc_csc, inf_error)
        - L_csc: Lower triangular matrix L in csc format
        - U_csc: Upper triangular matrix U in csc format
        - Pr_csc: Row permutation matrix in csc format
        - Pc_csc: Column permutation matrix in csc format
        - inf_error: Error measure from rank revealing LU
    """    
    # TODO...
    # PRBOLEM: SuperLU in Scipy only takes square matrix as input!     
    return

def simu_spPrrldu(M: csc_matrix, cutoff: float = 0.0, 
              maxdim: int = np.iinfo(np.int32).max
              ) -> Tuple[csc_matrix, csc_matrix, csc_matrix,
                         csc_matrix, csc_matrix, float]:
    """
    Simulator of partial rank revealing LDU decomposition
    Simulating the sparse operators in LU as there are no
    LU methods handling NON-SQUARE matrices in Scipy/...
    Args:
        M: Input csc sparse matrix 
        cutoff: truncation threshold
        maxdim: maximum dimension of factors
    Returns:
        Tuple containing (L_csc, U_csc, Pr_csc, Pc_csc, inf_error)
        - L_csc: Lower triangular matrix L in csc format
        - D_csc: Diagonal matrix D in csc format
        - U_csc: Upper triangular matrix U in csc format
        - Pr_csc: Row permutation matrix in csc format
        - Pc_csc: Column permutation matrix in csc format
        - inf_error: Error measure from rank revealing LU
    """
    assert maxdim > 0, "maxdim must be positive"
    # splu simulation
    try:
        dense_M = M.toarray()
    except AttributeError:
        dense_M = M.todense()
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

def spInterpolative_prrldu(M: csc_matrix, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1
                           ) -> Tuple[csc_matrix, csc_matrix]:
    """
    Compute sparse interpolative decomposition (ID) from sparse PRRLDU for sparse M.
    Args:
        M: Input sparse matrix
        **kwargs: Additional keyword arguments passed to prrldu
    Returns:
        Tuple containing (C, Z, pivot_columns, inf_error)
        - C: CSC-format sparse matrix containing selected columns 
        - Z: CSC-format sparse interpolation matrix
    """   
    L_csc, D_csc, U_csc, Pr_csc, Pc_csc, inf_error = simu_spPrrldu(M, cutoff, maxdim)
    k = D_csc.shape[0]
    U11 = csc_col_select(U_csc, k)   # U11 = U[:, :k], extract relevant submatrices
    iU11 = spsolve_triangular(U11, np.eye(U_csc.shape[0]), lower=False)  # Compute inverse of U11 through backsolving
    ZjJ = csc_matrix(iU11 @ U_csc.toarray())  # Compute interpolation matrix
    CIj = L_csc.dot(D_csc.dot(U11))  # Compute selected columns 
    C = Pr_csc.dot(CIj)              # Apply row permutation to get C
    Z = ZjJ.dot(Pc_csc.transpose())  # Apply column permutation to get Z
    return C, Z

def econSpInterpolative_prrldu(M: csc_matrix, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1
                           ) -> Tuple[csc_matrix, np.array, np.array]:
    """
    Compute sparse interpolative decomposition (ID) from sparse PRRLDU for sparse M.
    The storage of factor Z is further compressed in a more economic way, 
    stored by a dense coefficient matrix and a permutation vector
    Args:
        M: Input sparse matrix
        **kwargs: Additional keyword arguments passed to prrldu
    Returns:
        Tuple containing (C, Z, pivot_columns, inf_error)
        - C: CSC-format sparse matrix containing selected columns 
        - ZjJ: Dense matrix containing coefficients
        - pz: Permutation vector for the coefficient matrix 
    """   
    # Simulate the sparse prrldu decomposition (as there is no proper sparse prrldu for python so far)
    # Note: we actually do not need L
    try:
        dense_M = M.toarray()
    except AttributeError:
        dense_M = M.todense()
    L, d, U, pr, pc, inf_error = prrldu(dense_M, cutoff, maxdim)
    U_csc = csc_matrix(U)
    k = len(d)
    
    # Compute C (to modify...)
    C = csc_matrix(M.toarray()[:, pc[0:k]])  # TODO Note: we need to modify this to a sparse function later!
    
    # Computation of coefficients      
    U11 = csc_col_select(U_csc, k)  # Extract the relevant submatrix
    iU11 = spsolve_triangular(U11, np.eye(U_csc.shape[0]), lower=False)  # Compute the inverse of U11 through the sparse triangular solver
    ZjJ = iU11 @ U_csc.toarray()[:, k:]  # Compute interpolation coefficients
    return C, ZjJ, pc

# ! Problem of prrldu2 for sparse matrices !
def unit_test_1():
    # unit test for prrldu (dense & sparse)
    print("Unit test for dense/sparse prrldu starts!")
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
    print(f"prrldu (dense): Absolute max error = {max_err}")
    
    # Dense prrldu2 (prrldu is the normal plu with rank truncation but without column pivots)
    P, L, U = prrldu2(M_dense, 1e-10, min(m,n))
    Recon = P @ L @ U
    max_err = np.max(np.abs(Recon - M_dense))
    print(f"prrldu2 (dense): Absolute max error = {max_err}")

    # Sparse prrldu (fake simulation)
    L_csc, D_csc, U_csc, Pr_csc, Pc_csc, inf_error = simu_spPrrldu(M, 1e-10, min(m,n))
    recon = L_csc.dot(D_csc.dot(U_csc))
    recon_recover_rc = Pr_csc.dot(recon.dot(Pc_csc.transpose()))
    max_err = np.max(np.abs(recon_recover_rc.toarray() - M_dense))
    print(f"prrldu (sparse): Absolute max error = {max_err}")
    print("Unit test ends!")
    return 

def unit_test_2():
    # unit test for sparse interpolative decomposition
    print("Unit test for sparse interpolative decomposition starts!")
    rd.seed(10)
    m = 6
    r = 2
    n = 5
    A = random(m, r, density=0.5, format='csc', random_state=15)
    B = random(r, n, density=0.5, format='csc', random_state=30)
    M = A.dot(B)
    M_dense = M.toarray()
    C, Z = spInterpolative_prrldu(M, 1e-10, min(m,n))
    error = np.linalg.norm(M_dense - C.dot(Z).toarray(), ord='fro') / np.linalg.norm(M_dense, ord='fro')   
    print(f"Relative error: {error}")
    
    C, ZjJ, pc = econSpInterpolative_prrldu(M, 1e-10, min(m,n))
    
    print("Unit test ends!")
    return

#unit_test_1()
unit_test_2()

# A rule of thumb: coefficient matrix ZjJ is usually dense. 
# But the sparsity of M will lead to sparsity of ZjJ 
# TO BE VERIFIED!