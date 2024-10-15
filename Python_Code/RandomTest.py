import numpy as np
import sparse as sp
import tensorly as tl
from tensorly.contrib.decomposition import tensor_train_cross

from STTSVD import ttsvd  ## TT-SVD written by MENG
from Utils import TensorSparseStat

'''
def TestCase1():
    # Generate the random sparse tensor
    shape = [4, 4, 5, 3]
    sparsity = 0.8
    seed = 10
    SpT = sp.random(shape, 1-sparsity, random_state=seed)
    SpTd = SpT.todense()
# For TTSVD, how large rank should be exactly to decompose?

    rank = [1, 4, 15, 3, 1]
    factors = tl.decomposition.tensor_train(SpTd, rank)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(reconT - SpTd, 2)/tl.norm(SpTd, 2)}")

    rank = [1, 4, 14, 3, 1]
    tolerance = 1e-4
    maxiter = 100
    seed = 10
    factors = sparse_ttcross(SpT, rank, tolerance, maxiter, seed)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTCross reconstruction error is {tl.norm(reconT - SpT, 2) / tl.norm(SpT, 2)}")    
'''

def TestCase1():
    # Generate the random sparse tensor
    shape = [5, 8, 5, 3, 4]
    density = 0.05
    seed = 10
    SpT = sp.random(shape, density, random_state=seed)
    SpTd = SpT.todense()
    TensorSparseStat([SpTd])

    maxdim = 25
    cutoff = 2.7E-1    
    factors = ttsvd(SpTd, maxdim, cutoff)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(factors)
    
    rank = [1]
    for i in range(len(factors)-1):
        shape = factors[i+1].shape
        rank.append(shape[0])
        pass
    rank.append(1)
    print(f"The optimal rank is {rank}")
    
    spFactors = tensor_train_cross(SpTd, rank, cutoff)
    reconT = tl.tt_to_tensor(spFactors)
    print(f"The TTCross reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(spFactors)
    
    return 

TestCase1()
print("All unit tests finished.")