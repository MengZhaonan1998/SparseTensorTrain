import numpy as np
import sparse as sp
import scipy as sc

import tensorly as tl
from tensorly.contrib.decomposition import tensor_train_cross
from tensorly.decomposition import tensor_train

from STTCross import sparse_ttcross
from tensortrain_decomposition import TT_SVD, TT_IDscatter
from Utils import TensorSparseStat

def testCase1():
    print("Unit test 1 starts!")
    
    rank = [1, 30, 30, 1]        # TT rank
    order = [50, 50, 50]         # tensor order
    density = [0.1, 0.1, 0.1]    # density for every factor
    seed = [1, 2, 3]             # random seeds
    factors = []                 # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.random(shape)
        randl = np.random.random(shape)
        factor = np.where(randl>density[i], 0, factor)
        factors.append(factor)                
    SpTd = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    TensorSparseStat([SpTd])
    
    '''
    outputFlag = 0
    if outputFlag == 1:     
        tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
        tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
        cntData = SpT.nnz
        nnzData = SpT.data
        coord = SpT.coords
        with open(tnsPath, "w") as f:
            for i in range(cntData):
                f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {nnzData[i]}\n")
    '''

    # TT-SVD
    rank_max = max(rank)
    eps = 1e-10
    factors = TT_SVD(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT-ID
    rank_max = max(rank)
    eps = 1e-10
    factors = TT_IDscatter(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
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
    
    rank = [1, 5, 10, 5, 1]         # TT rank
    order = [10, 10, 10, 10]        # tensor order
    density = [0.1, 0.1, 0.1, 0.1]  # density for every factor
    seed = [1, 2, 3, 4]             # random seeds
    factors = []                    # factor list 

    # Construct sparse tensor factors in a sparse format
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.random(shape)
        randl = np.random.random(shape)
        factor = np.where(randl>density[i], 0, factor)
        factors.append(factor)                
    SpTd = tl.tt_to_tensor(factors)      # Tensorly supports sparse backends
    TensorSparseStat([SpTd])
    
    '''
    outputFlag = 0
    if outputFlag == 1:     
        tnsName = "syn_order_" + "_".join(map(str, order)) + "_synrank_" + "_".join(map(str, rank)) + ".tns"
        tnsPath = "/home/mengzn/Desktop/TensorData/" + tnsName
        cntData = SpT.nnz
        nnzData = SpT.data
        coord = SpT.coords
        with open(tnsPath, "w") as f:
            for i in range(cntData):
                f.write(f"{coord[0][i]} {coord[1][i]} {coord[2][i]} {nnzData[i]}\n")
    '''

    # TT-SVD
    rank_max = max(rank)
    eps = 1e-8
    factors = TT_SVD(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-SVD is {error}")
    TensorSparseStat(factors)
    
    # TT-ID
    rank_max = max(rank)
    eps = 1e-8
    factors = TT_IDscatter(SpTd, rank_max, eps)
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-ID is {error}")
    TensorSparseStat(factors)
    
    # TT Cross 
    random_state = 10
    tol = 1e-8
    maxiter = 100
    #factors = sparse_ttcross(SpTd, rank, tol, maxiter, random_state)
    factors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)
    # Check the reconstruction error and sparsity information
    reconT = tl.tt_to_tensor(factors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error of TT-Cross is {error}")
    TensorSparseStat(factors)     
    
    print("Unit test 1 ends!\n")
    return

#testCase1()
testCase2()
