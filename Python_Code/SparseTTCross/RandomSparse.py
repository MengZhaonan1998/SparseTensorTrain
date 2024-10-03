import numpy as np
import sparse as sp
import tensorly as tl
from SparseTTCross import sparse_ttcross
from tensorly.contrib.decomposition import tensor_train_cross

def testCase1():
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
    
testCase1()
pass
