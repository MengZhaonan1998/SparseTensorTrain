import numpy as np
import sparse as sp
import tensorly as tl
from TTCrossAlgo import tensor_train_cross

def testCase1():
    # Generate the random sparse tensor
    shape = [15, 10, 20, 13]
    sparsity = 0.01
    seed = 2
    SpT = sp.random(shape, 1-sparsity, random_state=seed)

    # Temporarily convert the sparse tensor into dense format, 
    # as now we do not have tensor train algorithms taking sparse formats as input
    SpTd = SpT.todense()

    rank = [1, 60, 60, 60, 1]
    factors = tl.decomposition.tensor_train(SpTd, rank)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(reconT - SpTd, 2)/tl.norm(SpTd, 2)}")

    #rank = [1, 5, 8, 5, 1]
    #tolerance = 1e-5
    #maxiter = 1000
    #seed = 1
    #factors = tensor_train_cross(SpTd, rank, tolerance, maxiter, seed)
    #reconT = tl.tt_to_tensor(factors)
    #print(f"The TTCross reconstruction error is {tl.norm(reconT - SpTd, 2) / tl.norm(SpTd, 2)}")
    
testCase1()
pass
