import numpy as np
import sparse as sp
import scipy as sc

import tensorly as tl
from tensorly.contrib.decomposition import tensor_train_cross
from tensorly.decomposition import tensor_train

from STTCross import sparse_ttcross
from STTSVD import ttsvd
from Utils import TensorSparseStat


def testCase1():
    '''
    20x20x20 tensor (8000 size) -> 20x10 + 10x20x10 + 20x10 factors (2400 size)
    Sparse ...
    '''
    print("Unit test 1 starts!")
    
    rank = [1, 10, 10, 1]       # TT rank
    order = [20, 20, 20]        # tensor order
    density = [0.1, 0.1, 0.1]   # density for every factor
    seed = [1, 2, 3]            # random seeds
    factors = []                # factor list 

    # Construct sparse tensor factors in a sparse format
    #rng = np.random.RandomState(2)
    #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density = density[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    SpTd = SpT.todense()
    TensorSparseStat([SpTd])
    
    # TT SVD
    factors = tensor_train(SpTd, rank)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    #factors = sparse_ttcross(SpTd, rank, tol, maxiter, random_state)
    factors = tensor_train_cross(SpTd, [1,9,9,1], tol, maxiter, random_state)
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)     
    
    print("Unit test 1 ends!\n")
    return

def testCase2():
    '''
    10x20x20x10 tensor (40000 size) -> 5x10 + 5x20x10 + 5x20x10 + 5x10 factors (2100 size)
    Sparse ...
    '''
    print("Unit test 2 starts!")
    
    rank = [1, 5, 10, 5, 1]         # TT rank
    order = [10, 20, 20, 10]        # tensor order
    density = [0.2, 0.1, 0.1, 0.2]  # density for every factor
    seed = [1, 2, 3, 4]             # random seeds
    factors = []                    # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=density[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    SpTd = SpT.todense()
    TensorSparseStat([SpTd])

    # TT SVD
    factors = tensor_train(SpTd, rank)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)

    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    factors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)    
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)
    
    print("Unit test 2 ends!\n")
    return
 
    
def testCase3():
    '''
    Same configuration with the Julia unit test of TTID
    To compare ...
    '''
    print("Unit test 3 starts!")
    
    rank = [1, 5, 9, 7, 8, 1]           # TT rank
    order = [3, 6, 2, 4, 5]             # tensor order
    density = [0.2, 0.1, 0.2, 0.1, 0.2] # density for every factor
    seed = [1, 2, 3, 4, 5]              # random seeds
    factors = []                        # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=density[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    SpTd = SpT.todense()
    TensorSparseStat([SpTd])
    
    # TT SVD
    factors = tensor_train(SpTd, rank)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT Cross 
    random_state = 100
    tol = 1e-5
    maxiter = 100
    factors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)     
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)
    
    print("Unit test 3 ends!\n")
    return


testCase1()
#testCase2()
#testCase3()

# NOTE:
# control variable: Dimension, shape...?