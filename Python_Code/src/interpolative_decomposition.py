import numpy as np
from scipy.linalg import solve, qr

from scipy.linalg.interpolative import interp_decomp

# To be updated ... 
# Other variants of ID
# More unit tests

def interpolative_nuclear(M, cutoff=0.0, maxdim=None):
    '''
    Interpolative Decomposition (Nuclear)
    M = C * X
    cutoff - truncation threshold
    maxdim - maximum rank
    '''
    if maxdim is None:
        maxdim = min(M.shape)
    
    maxdim = min(maxdim, M.shape[0], M.shape[1])
    cols = []
    K = M.T @ M
    m = K.shape[0]
    Kt = K
    error = 0.0

    for t in range(maxdim):
        Kt2 = Kt @ Kt
        # Select the column with the maximum score
        l = max((p for p in range(m) if p not in cols), key=lambda p: Kt2[p, p] / Kt[p, p])
        max_err2 = Kt2[l, l] / Kt[l, l]
        cols.append(l)
        error = np.sqrt(np.abs(max_err2))
        if max_err2 < cutoff**2:
            break
        # Schur complement step
        Kt = K - K[:, cols] @ solve(K[np.ix_(cols, cols)], K[cols, :])
    
    # C selection    
    C = M[:, cols]
    
    # X = C \ M
    X = np.zeros([len(cols), M.shape[1]])
    qc, rc = qr(C,mode='economic')
    for i in range(M.shape[1]):
        m = M[:, i]
        z = qc.T @ m
        X[:,i] = solve(rc, z)
     
    # Enforce interpolation structure
    for w in range(len(cols)):
        for r in range(len(cols)):
            X[r, cols[w]] = 1.0 if r == w else 0.0

    return C, X, cols, error

def unit_test_1():
    # Test of a small low-rank fixed matrix
    print("Unit test 1 starts!")
    A = np.array([[3,1],[8,2],[9,-5],[-7,4]])
    B = np.array([[4,6,2],[8,-1,-4]])    
    M = A @ B
    
    maxdim = 2
    cutoff = 1e-4
    C, X, cols, error = interpolative_nuclear(M, cutoff, maxdim)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 1 ends!")
    return

def unit_test_2():
    # Test of a small low-rank random matrix 
    print("Unit test 2 starts!")
    m = 4
    n = 6
    rank = 3
    A = np.random.random((m,rank))
    B = np.random.random((rank,n))
    M = A @ B
    
    maxdim = 3
    cutoff = 1e-4
    C, X, cols, error = interpolative_nuclear(M, cutoff, maxdim)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 2 ends!")
    return

def unit_test_3():
    # Test of a small full-rank random matrix
    print("Unit test 3 starts!")
    m = 12
    n = 11
    M = np.random.random((m,n))

    cutoff = 1
    C, X, cols, error = interpolative_nuclear(M, cutoff)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 3 ends!")
    return

def unit_test_4():
    # Test of a small full-rank random matrix
    print("Unit test 4 starts!")
    m = 12
    n = 5
    M = np.zeros((m,n))
    M[1,2] = 1
    M[3,4] = 1

    cutoff = 1
    C, X, cols, error = interpolative_nuclear(M, cutoff)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 4 ends!")
    return

#unit_test_1()
#unit_test_2()
#unit_test_3()
unit_test_4()