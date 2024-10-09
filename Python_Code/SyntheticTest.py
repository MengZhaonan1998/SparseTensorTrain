import numpy as np
import tensorly as tl
import sparse as sp
import scipy as sc

from tensorly.contrib.decomposition import tensor_train_cross
from STTCross import sparse_ttcross
from STTSVD import ttsvd

def SparseInfo(factors: list[np.array]):
    avgsparsity = 0
    avgdensity = 0
    nlist = len(factors)
    for i in range(nlist):
        factor = factors[i]
        cntzero = np.count_nonzero(np.abs(factor) == 0.0)
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

def testCase1():
    '''
    20x20x20 tensor (8000 size) -> 20x10 + 10x20x10 + 20x10 factors (2400 size)
    Sparse ...
    '''
    rank = [1, 10, 10, 1]       # TT rank
    order = [20, 20, 20]        # Tensor order
    sparsity = [0.9, 0.9, 0.9]  # Sparsity for every factor
    seed = [1, 2, 3]           # Random seeds
    factors = []                # factor list 

    # Construct sparse tensor factors in a sparse format
    #rng = np.random.RandomState(2)
    #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=1-sparsity[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    print(f"The density of the synthetic sparse tensor SpT is {SpT.density}. The number of non-zeros is {SpT.nnz}")

    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    spFactors = sparse_ttcross(SpT, rank, tol, maxiter, random_state)    

    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpT, 2) / tl.norm(SpT, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    for i in range(len(spFactors)):
        print(f"Factor {i}: nnz = {spFactors[i].nnz}, density = {spFactors[i].density}, sparsity = {1-spFactors[i].density}")
        

def testCase2():
    '''
    10x20x20x10 tensor (4e4 size) -> 5x10 + 5x20x10 + 5x20x10 + 5x10 factors (2100 size)
    Sparse ...
    '''
    rank = [1, 5, 10, 5, 1]       # TT rank
    order = [10, 20, 20, 10]        # Tensor order
    sparsity = [0.8, 0.9, 0.9, 0.8]  # Sparsity for every factor
    seed = [1, 2, 3, 4]           # Random seeds
    factors = []                # factor list 

    # Construct sparse tensor factors in a sparse format
    #rng = np.random.RandomState(2)
    #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=1-sparsity[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    print(f"The density of the synthetic sparse tensor SpT is {SpT.density}. The number of non-zeros is {SpT.nnz}")

    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    spFactors = sparse_ttcross(SpT, rank, tol, maxiter, random_state)    

    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpT, 2) / tl.norm(SpT, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    for i in range(len(spFactors)):
        print(f"Factor {i}: nnz = {spFactors[i].nnz}, density = {spFactors[i].density}, sparsity = {1-spFactors[i].density}")
  
    
def testCase3():
    '''
    10x20x20x10 tensor (4e4 size) -> 5x10 + 5x20x10 + 5x20x10 + 5x10 factors (2100 size)
    Sparse ...
    '''
    rank = [1, 3, 5, 5, 3, 1]       # TT rank
    order = [5, 10, 10, 10, 5]        # Tensor order
    sparsity = [0.6, 0.8, 0.9, 0.8, 0.6]  # Sparsity for every factor
    seed = [1, 2, 3, 4, 5]           # Random seeds
    factors = []                # factor list 

    # Construct sparse tensor factors in a sparse format
    #rng = np.random.RandomState(2)
    #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    for i in range(len(order)):
        shape = (rank[i], order[i], rank[i+1])
        factor = sp.random(shape, density=1-sparsity[i], random_state=seed[i])
        factors.append(factor)                
    SpT = tl.tt_to_tensor(factors)            # Tensorly supports sparse backends
    print(f"The density of the synthetic sparse tensor SpT is {SpT.density}. The number of non-zeros is {SpT.nnz}")

    # TT Cross 
    random_state = 100
    tol = 1e-10
    maxiter = 500
    spFactors = sparse_ttcross(SpT, rank, tol, maxiter, random_state)    
    
    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpT, 2) / tl.norm(SpT, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    for i in range(len(spFactors)):
        print(f"Factor {i}: nnz = {spFactors[i].nnz}, density = {spFactors[i].density}, sparsity = {1-spFactors[i].density}")
  

#testCase1()
#testCase2()
testCase3()

# NOTE:
# control variable: Dimension, shape...?