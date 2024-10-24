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
    print("Unit test 1 starts!")
    
    rank = [1, 30, 30, 1]       # TT rank
    order = [50, 50, 50]        # tensor order
    density = [0.01, 0.01, 0.01]   # density for every factor
    seed = [13, 0, 3033]            # random seeds
    factors = []                # factor list 

    # Construct sparse tensor factors in a sparse format
    #rng = np.random.RandomState(2)
    #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density = density[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
         
    tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
    tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
    cntData = SpT.nnz
    nnzData = SpT.data
    coord = SpT.coords
    with open(tnsPath, "w") as f:
        for i in range(cntData):
            f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {nnzData[i]}\n")
    
    SpTd = SpT.todense()
    TensorSparseStat([SpTd])
    
    # TT SVD
    rank_max = max(rank)
    eps = 1e-10
    factors = ttsvd(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT Cross 
    random_state = 10
    tol = 1e-10
    maxiter = 500
    rank = [1, 30, 30, 1]
    #factors = sparse_ttcross(SpTd, rank, tol, maxiter, random_state)
    factors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)     
    
    print("Unit test 1 ends!\n")
    return

def testCase2():
    print("Unit test 2 starts!")
    
    rank = [1, 50, 200, 50, 1]         # TT rank
    order = [50, 50, 50, 50]        # tensor order
    density = [0.01, 0.01, 0.01, 0.01]  # density for every factor
    seed = [1, 2, 3, 4]             # random seeds
    factors = []                    # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=density[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    
    tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
    tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
    cntData = SpT.nnz
    nnzData = SpT.data
    coord = SpT.coords
    with open(tnsPath, "w") as f:
        for i in range(cntData):
            f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {coord[3][i]} {nnzData[i]}\n")
    
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
    rank = [1, 20, 20,20,1]
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

#testCase1()
testCase2()
#testCase3()