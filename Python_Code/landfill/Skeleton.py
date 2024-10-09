import numpy as np
import random as rd
import scipy as sc

def SkeletonDecomp(W: np.array, rank: int) -> list[np.array]:
    row = W.shape[0]
    col = W.shape[1]
    C = np.zeros([row, rank])
    A = np.zeros([rank, rank])
    R = np.zeros([rank, col])
    v_row = np.zeros([rank])
    v_col = np.zeros([rank])
    while (np.linalg.matrix_rank(A) != rank):
        v_row = rd.sample(range(row), rank)
        v_col = rd.sample(range(col), rank)
        for i in range(rank):
            for j in range(rank):
                A[i,j] = W[v_row[i], v_col[j]]
    C = W[:, v_col]
    R = W[v_row, :]
    invA = np.linalg.inv(A)
    return [C, invA, R]

def SparseSkeletonDecomp(SpW):  # SpW should be COO sparse format from scipy.sparse 
    row = SpW.shape[0]
    col = SpW.shape[1]
    coo_rowi = SpW.coords[0]
    coo_coli = SpW.coords[1]
    #entry = SpW.data

    # If we are aware of the rank of a SPARSE matrix ...
    rough_rank = len(set(coo_rowi))   # Roughly estimate the rank (maybe not correct...)
    
    # For testing, dense format is still used temporarily
    C = np.zeros([row, rough_rank])
    A = np.zeros([rough_rank, rough_rank])
    R = np.zeros([rough_rank, col])
    #v_row = np.zeros([rough_rank])
    v_col = np.zeros([rough_rank])
    v_row = np.unique(coo_rowi)
    for i in range(rough_rank):
        argi = np.where(coo_rowi==v_row[i])
        if len(argi[0]) == 1:
            v_col[i] = coo_coli[argi]
        else:
            for j in range(len(argi)):
                a = np.where(argi==coo_coli[argi[j]])
                if a[0].size == 0:
                    v_col[i] = coo_coli[argi[0][j]]
    v_col = v_col.astype('int32')
    
    W = SpW.toarray()
    for i in range(rough_rank):
        for j in range(rough_rank):
            A[i,j] = W[v_row[i], v_col[j]]

    C = W[:, v_col]
    R = W[v_row, :]
    invA = np.linalg.inv(A)
    return [C, invA, R]

# an artificial sparse matrix
W = np.array([[1, 0, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 0, 0, 5],
                [0, 0, 7, 0, 0]])
C, invA, R = SkeletonDecomp(W, 3)
print(C @ invA @ R)

# Suppose the rank is awared... (by guess?)
n=100
density = 0.1
SpM = sc.sparse.random(n, n, density, 'coo')
coo_rowi = SpM.coords[0]
coo_coli = SpM.coords[1]
entry = SpM.data
#print(f"The sparse matrix SpM is given by: \n{SpM.toarray()}")

# Roughly estimate the rank
distinct = set(coo_rowi)
rough_rank = len(distinct)

[C, invA, R] = SparseSkeletonDecomp(SpM)
print(C @ invA @ R - SpM.toarray())

pass