import numpy as np
import sparse as sp
import numpy.linalg as la
import tensorly as tl

def ttsvd(tensorX: tl.tensor, r_max: int, eps: float) -> list[tl.tensor]:
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
    delta = (eps / np.sqrt(dim - 1)) * tl.norm(tensorX, 2)  # Truncation parameter
    
    W = tensorX  # Copy tensor X -> W
    nbar = W.size   # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing tt factors
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    
    for i in iterlist:
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])  # Reshape W
        U, S, Vh = la.svd(W)  # SVD of W matrix
        # Compute rank r
        s = 0
        j = S.size 
    
        while s <= delta * delta:  # r_delta_i = min(j:sigma_j+1^2 + sigma_j+2^2 + ... <= delta^2)
            j -= 1
            s += S[j] * S[j]
        j += 1
        ri = min(j, r_max)  # r_i-1 = min(r_max, r_delta_i)
    
        Ti = tl.reshape(Vh[0:ri, :], [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri  # Renewal r
        W = U[:, 0:ri] @ np.diag(S[0:ri])  # W = U[..] * S[..]
        ttList.append(Ti)  # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList

'''
def TestCase1():
    # Complete random tensor test 1 (TTSVD)
    print("Unit test 1 starts!")
    
    shape = [3, 4, 5, 3, 2]
    density = 0.1
    seed = 100
    SpT = sp.random(shape, density, random_state=seed)
    SpTd = SpT.todense()  
    TensorSparseStat([SpTd])
    
    factors = TTSVD(SpTd, 12, 1e-4)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(factors)
    
    print("Unit test 1 ends!\n")
    return

def TestCase2():
    # Complete random tensor test 1 (TTSVD)
    print("Unit test 2 starts!")
    
    shape = [5, 2, 6, 7, 3]
    density = 0.05
    seed = 200
    SpT = sp.random(shape, density, random_state=seed)
    SpTd = SpT.todense()  
    TensorSparseStat([SpTd])
    
    factors = TTSVD(SpTd, 21, 1e-4)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(factors)
    
    print("Unit test 2 ends!\n")
    return

TestCase1()
TestCase2()
print("All unit tests finished.")
'''

'''
rng = tl.check_random_state(10)  # Fix the random seed for reproducibility
shapeX = [5, 3, 7, 8, 4]         # Dimension of the tensor X                         
tensorX = T.tensor(rng.normal(size=shapeX, loc=0, scale=1), dtype='float32')  # Generate a random tensor X

ttFactors = TTSVD(tensorX, 10, 1e-3)
ttFactors.reverse()
reconX = tl.tt_to_tensor(ttFactors)
error = tl.norm(tensorX - reconX, 2)/tl.norm(tensorX, 2)
print(error)
'''

'''
import julia
julia.install()
from julia.api import Julia
jl=Julia(compiled_modules=False)
from julia import ITensorTCI
ii = ITensorTCI.interpolative
M = [[1,2],[3,4]]
ii(M,1)
'''