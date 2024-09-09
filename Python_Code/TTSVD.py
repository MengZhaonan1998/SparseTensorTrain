import numpy as np
import numpy.linalg as la
import tensorly as tl
import tensorly.tenalg.proximal as prox


def TTSVD(tensorX: tl.tensor, r_max: float, eps: float) -> list[tl.tensor]:
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
    return ttList

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

shape = [50, 50, 50, 50]
rank = [1, 4, 4, 4, 1]
dense_factors = tl.random.random_tt(shape, rank, False, 0)  # Factor initialization

sparse_factors = []
for i in range(len(shape)):
    dense_factor = dense_factors[i]
    sparse_factor = prox.proximal_operator(dense_factor, normalized_sparsity=50)  # Sparsity introduction
    sparse_factors.append(sparse_factor)

U, S, Vh = np.linalg.svd(sparse_factors[0])

pass