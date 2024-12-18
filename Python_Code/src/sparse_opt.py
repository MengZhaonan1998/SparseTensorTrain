import numpy as np
import random as rd
from scipy.sparse import csc_matrix, random
from scipy.sparse.linalg import splu
from scipy.linalg import lu
from typing import Tuple, Union, List
from interpolative_decomposition import prrldu2, prrldu

def permutation_vector_to_matrix(p):
    """
    Converts a permutation vector to a permutation matrix.
    Args: p: A permutation vector (list or numpy array).
    Returns: A permutation matrix (numpy array).
    """
    n = len(p)
    matrix = np.zeros((n, n), dtype=int)
    for i, j in enumerate(p):
        matrix[i, j] = 1
    return matrix

def csc_row_select(A: csc_matrix, row_s: int) -> csc_matrix:
    # SPARSE A -> B = A[0: row_s, :]
    nz_data = A.data      # non-zero elements
    row_index = A.indices # row indices 
    col_offset = A.indptr # column offset
    rows = A.shape[0]     # row number
    cols = A.shape[1]     # column number 
    assert row_s <= rows, "row_s cannot be larger than number of rows"
 
    # Initialization
    new_nz_data = np.zeros(len(nz_data))
    new_row_index = np.zeros(len(nz_data))
    new_col_offset = np.zeros(cols + 1)
    
    # Slicing
    cnt = 0
    col_offset_ptr = 0
    for i in range(len(nz_data)):
        if i == col_offset[col_offset_ptr]:
            new_col_offset[col_offset_ptr] = cnt
            col_offset_ptr += 1
            while col_offset[col_offset_ptr] == col_offset[col_offset_ptr-1]:
                new_col_offset[col_offset_ptr] = cnt
                col_offset_ptr += 1        
        if row_index[i] < row_s:
            new_nz_data[cnt] = nz_data[i]
            new_row_index[cnt] = row_index[i]
            cnt += 1    
    new_nz_data = new_nz_data[0: cnt]
    new_row_index = new_row_index[0: cnt]    
    new_col_offset[cols] = cnt    
    
    # Construct the new csc matrix
    B = csc_matrix((new_nz_data, new_row_index, new_col_offset), shape=(row_s, cols))    
    return B

def csc_col_select(A: csc_matrix, col_s: int) -> csc_matrix:
    # SPARSE A -> B = A[:, 0: col_s]
    nz_data = A.data      # non-zero elements
    row_index = A.indices # row indices 
    col_offset = A.indptr # column offset
    assert col_s < len(col_offset), "col_s cannot be larger than number of columns"
    
    # Get new non-zero data
    trunc_nz_idx = col_offset[col_s]
    new_nz_data = nz_data[0: trunc_nz_idx]
    
    # Get new row indices
    new_row_index = row_index[0: trunc_nz_idx]
    
    # Get new column offset
    new_col_offset = np.zeros(col_s + 1)
    new_col_offset[0: col_s] = col_offset[0: col_s]
    new_col_offset[col_s] = trunc_nz_idx
    
    # Construct the new csc matrix
    row = A.shape[0]      
    B = csc_matrix((new_nz_data, new_row_index, new_col_offset), shape=(row, col_s))
    return B

def test_splu():
    # Test of superlu for sparse LU decomposition
    # But the problem is it can only handle square matrices
    A = csc_matrix([[1,2,0,4], [1,0,0,1], [1,0,2,1], [2,2,1,0.]])
    lu = splu(A)

    print(f"Row permutation: {lu.perm_r}")
    print(f"Column permutation: {lu.perm_c}")
    print(f"L matrix: {lu.L.toarray()}")
    print(f"U matrix: {lu.U.toarray()}")
    
    Pr = csc_matrix((np.ones(4), (lu.perm_r, np.arange(4))))
    Pc = csc_matrix((np.ones(4), (np.arange(4), lu.perm_c)))
    
    recon_A = (Pr.T @ (lu.L @ lu.U) @ Pc.T)
    print(f"Reconstructed A: {recon_A.toarray()}")
    return

def unit_test_1():
    A = csc_matrix([[1,0,0,4], [1,0,0,-3], [0,0,0,0], [2,0,1,0], [-1,0,8,5]])
    col_s = 2
    row_s = 3
    B = csc_col_select(A, col_s)
    C = csc_row_select(A, row_s)
    print(f"A[:,:] =\n {A.toarray()}")
    print(f"A[:,0:{col_s}] =\n {B.toarray()}")
    print(f"A[0:{row_s},:] =\n {C.toarray()}")
    
    A = csc_matrix([[1,0,0,1,0], [2,0,0,0,-5], [1,0,3,0,-1], [2,0,-2,6,0]])
    col_s = 3
    row_s = 3
    B = csc_col_select(A, col_s)
    C = csc_row_select(A, row_s)
    print(f"A[:,:] =\n {A.toarray()}")
    print(f"A[:,0:{col_s}] =\n {B.toarray()}")
    print(f"A[0:{row_s},:] =\n {C.toarray()}")
    return