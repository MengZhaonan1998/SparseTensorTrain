import numpy as np
import tensorly as tl
import sparse as sp
from tensorly.contrib.sparse import tt_to_tensor
from sparse_interpolative import spInterpolative_prrldu
from sparse_opt import csc_row_select, csc_col_select

def RandomSpTT(shape: list, rank: list, density: list, seed: list) -> list[sp.COO]:
    """
    Generate a random sparse tensor train
    Args:
        shape (list): tensor shape
        rank (list): tensor-train rank
        density (list): density for each TT factor
        seed (list): random seed
    """
    low = -2.0
    high = 3.0
    sptt = []
    for i in range(len(shape)):
        facShape = (rank[i], shape[i], rank[i+1])
        np.random.seed(seed[i])
        factor = np.random.uniform(low, high, size=facShape)
        randl = np.random.random(facShape)
        factor = np.where(randl>density[i], 0, factor)
        sptt.append(sp.COO(factor))                
    return sptt

def SpTT2Tensor(SpTT: list[sp.COO]) -> sp.COO:
    fullTT = []
    for i in range(len(SpTT)):
        fullTT.append(SpTT[i].todense())
    fullTensor = tl.tt_to_tensor(fullTT)
    spTensor = sp.COO(fullTensor)
    return spTensor

def SparseTT_IDPRRLDU(tensorX: sp.COO, r_max: int, eps: float, verbose: int = 0) -> list[sp.COO]:
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
    delta = (eps / np.sqrt(dim - 1)) * tl.norm(tensorX.todense(), 2)  # Truncation parameter
    
    W = tensorX        # Copy tensor X -> W
    nbar = W.size      # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing tt factors
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    
    for i in iterlist:       
        # Reshape W: ND-COO -> 2D-COO        
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])         
        
        # Sparse ID (W: 2D-COO) -> C, X (CSC)
        C, X = spInterpolative_prrldu(W, cutoff=delta, maxdim=r_max)    
        if verbose == 1:
            rerror = tl.norm(C.toarray() @ X.toarray() - W.todense(), 2) / tl.norm(W.todense(), 2)
            print(f"Iteration {i} -- low rank id approximation error = {rerror}")

        ri = C.shape[1]   # r_i-1 = min(r_max, r_delta_i)
        Ti = sp.COO(X)    #sp.COO(csc_row_select(X, ri))
        Ti = tl.reshape(Ti, [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri            # Renewal r
        W = csc_col_select(C, ri)
        W = sp.COO(W)     
        ttList.append(Ti) # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList

def toy_test():
    # Synthetic data generation
    shape = [8, 10, 10, 8]
    rank = [1, 6, 14, 7, 1]
    density = [0.3, 0.1, 0.1, 0.3]
    seed = [1, 2, 3, 4]
    sptt = RandomSpTT(shape, rank, density, seed)
    spTensor = SpTT2Tensor(sptt)
    
    # Sparse TT decomposition
    r_max = max(rank)
    cutoff = 1e-10
    sptt = SparseTT_IDPRRLDU(spTensor, r_max, cutoff, 1)
    ReconTensor = SpTT2Tensor(sptt)
    print(f"max error = {np.max(np.abs(ReconTensor.todense()-spTensor.todense()))}")
    return

toy_test()
