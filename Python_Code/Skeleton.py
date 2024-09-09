import numpy as np
import random as rd

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

# an artificial sparse matrix
W = np.array([[1, 0, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 0, 0, 5],
                [0, 0, 7, 0, 0]])
C, invA, R = SkeletonDecomp(W, 3)
print(C @ invA @ R)