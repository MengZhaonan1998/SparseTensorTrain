import numpy as np
import tensorly as tl
import random as rd

from tensorly.contrib.decomposition import tensor_train_cross
from tensorly.decomposition import tensor_train

def TT_Cross(T: tl.tensor, tol: float, iter_max: int):
    shape_T = T.shape   # Shape of the input tensor
    dim = len(shape_T)  # Number of dimensions(orders)
    iter = 0            # Number of iterations
    while iter < iter_max:
        iter += 1
        for k in range(dim):
            #TODO...
            pass
        for k in reversed(range(dim)):
            #TODO...
            pass
    
    
    
    return

def SparseInfo(factors: list[np.array]):
    avgsparsity = 0
    avgdensity = 0
    nlist = len(factors)
    for i in range(nlist):
        factor = factors[i]
        cntzero = np.count_nonzero(np.abs(factor) < 1e-15)
        size = factor.size
        sparsity = cntzero / size
        avgsparsity += sparsity
        density = 1 - sparsity
        avgdensity += density
        print(f"Tensor factor {i}: count of zero = {cntzero}, size = {size}, sparsity = {sparsity}, density = {density}")
    
    avgsparsity /= nlist
    avgdensity /= nlist
    print(f"Factors' average sparsity: {avgsparsity}; average density: {avgdensity}.")
    
    if nlist > 1:
        recon_tensor = tl.tt_to_tensor(factors)
        cntzero = np.count_nonzero(recon_tensor == 0)
        size = recon_tensor.size
        sparsity = cntzero / size
        density = 1 - sparsity
        print(f"Reconstruction tensor: count of zero = {cntzero}, size = {size}, sparsity = {sparsity}, density = {density}")
    return

rank = [1, 10, 10, 1]
order = [20, 20, 20]
sparsity = [0.90, 0.90, 0.90]
factors = []
for i in range(len(order)):
    N = rank[i] * order[i] * rank[i+1]
    factor = np.zeros(N)
    for j in range(N):
        r = rd.random()
        if r > sparsity[i]:
            factor[j] = rd.random()
    factor = tl.tensor(factor.reshape(rank[i], order[i], rank[i+1]))
    factors.append(factor)
SpT1 = tl.tt_to_tensor(factors)
SparseInfo([SpT1])
    
factors = tensor_train_cross(SpT1, rank)
rec_SpT1 = tl.tt_to_tensor(factors)
error = tl.norm(rec_SpT1 - SpT1)/tl.norm(SpT1)
SparseInfo(factors)

print(error)

