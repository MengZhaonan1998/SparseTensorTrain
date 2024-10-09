import numpy as np
import tensorly as tl
import tensorly.tenalg.proximal as prox
import matplotlib.pyplot as plt

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

def SparseFactorInfo(factors: list[np.array]):
    avgsparsity = 0
    avgdensity = 0
    nlist = len(factors)
    for i in range(nlist):
        factor = factors[i]
        cntzero = np.count_nonzero(factor == 0)
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

def RelaNormDiff(X1: np.array, X2: np.array) -> float:
    n1 = tl.norm(X1 - X2, 2)
    n2 = tl.norm(X2)
    return n1/n2

# Synthetic Sparse TT
# Factor initialization
shape = [30, 60, 90, 6]
rank = [1, 5, 5, 5, 1]
dense_factors = tl.random.random_tt(shape, rank, False, 0)  

# Sparsity introduction (synthetic sparse tensor factors)
sparse_factors = []
for i in range(len(shape)):
    dense_factor = dense_factors[i]
    sparse_factor = prox.proximal_operator(dense_factor, normalized_sparsity=100)  
    sparse_factors.append(sparse_factor)
SparseFactorInfo(sparse_factors)

# Construct a tensor with factors
full_tensor = tl.tt_to_tensor(sparse_factors)  
print(f"There are {full_tensor.size - np.count_nonzero(full_tensor == 0)} non-zeros in the full tensor")
print(f"The sparsity of the full tensor is {np.count_nonzero(full_tensor == 0)/full_tensor.size}")
train_factors = tl.decomposition.tensor_train(full_tensor, rank)
recon_tensor = tl.tt_to_tensor(train_factors)
error = RelaNormDiff(recon_tensor, full_tensor)  # Relative error between recon tensor and orig tensor
print(f"The relative norm difference between recon tensor and orig tensor is {error}")
SparseFactorInfo(train_factors)
print(f"The sparsity of the recon tensor is {np.count_nonzero(recon_tensor == 0)/recon_tensor.size}")

# Sparsity introduction to TT factors
sparse_train_factors = []
normlist = [200,600,600,200]
for i in range(len(shape)):
    train_factor = train_factors[i]
    sparse_train_factor = prox.proximal_operator(train_factor, normalized_sparsity=1000)  
    sparse_train_factors.append(sparse_train_factor)
recon_tensor = tl.tt_to_tensor(sparse_train_factors)
error = RelaNormDiff(recon_tensor, full_tensor)  # Relative error between recon tensor and orig tensor
print(f"The relative norm difference between recon tensor and orig tensor is {error}")
SparseFactorInfo(sparse_train_factors)



pass

sparsity_x = [0.992, 0.96, 0.92, 0.84, 0.76, 0.68, 0.6, 0.44, 0.2]
error_y = [1, 1.1, 0.88, 0.63, 0.51, 0.40, 0.31, 0.23, 0.19]

plt.figure()
plt.scatter(sparsity_x,error_y)
plt.xlabel(r"$AvgSparsity_{T''}$")
#plt.xticks(sparsity_x)
plt.yticks(error_y)
plt.ylabel(r"$error_{A''}$")
plt.grid()
plt.show()
