import numpy as np
import tensorly as tl
import sparse as sp
import scipy as sc
from TTCrossAlgo import tensor_train_cross

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

    # Temporarily convert sparse SpT into densor format, 
    # as ttcross api so far does not support sparse format
    SpTd = SpT.todense()

    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    spFactors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)    

    # Hard thresholding. Is this step necessary?
    for i in range(len(spFactors)):
        spFactors[i][np.abs(spFactors[i])<1e-10]=0.0

    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    SparseInfo(spFactors)

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

    # Temporarily convert sparse SpT into densor format, 
    # as ttcross api so far does not support sparse format
    SpTd = SpT.todense()

    # TT Cross 
    random_state = 1
    tol = 1e-10
    maxiter = 500
    spFactors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)    

    # Hard thresholding. Is this step necessary?
    for i in range(len(spFactors)):
        spFactors[i][np.abs(spFactors[i])<1e-10]=0.0

    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    SparseInfo(spFactors)
    
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

    # Temporarily convert sparse SpT into densor format, 
    # as ttcross api so far does not support sparse format
    SpTd = SpT.todense()

    # TT Cross 
    random_state = 100
    tol = 1e-10
    maxiter = 500
    spFactors = tensor_train_cross(SpTd, rank, tol, maxiter, random_state)    

    # Hard thresholding. Is this step necessary?
    for i in range(len(spFactors)):
        spFactors[i][np.abs(spFactors[i])<1e-10]=0.0

    # Check the reconstruction error
    reconT = tl.tt_to_tensor(spFactors)
    error = tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)
    print(f"The reconstruction error is {error}")

    # Check the sparsity of the tensor train
    SparseInfo(spFactors)

testCase1()
testCase2()
testCase3()