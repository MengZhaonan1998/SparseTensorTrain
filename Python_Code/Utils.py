import numpy as np

# Show sparsity information of the input tensor array 
def TensorSparseStat(factors: list[np.array]):
    print("The sparsity statistics of the input tensor is as follows ...")
    for i in range(len(factors)):
        factor = factors[i]
        size = factor.size
        shape = factor.shape
        cntzero = np.count_nonzero(np.abs(factor) < 1e-10)
        cntnzero = size - cntzero
        sparsity = cntzero / size
        density = cntnzero / size
        print(f"Tensor factor {i}: shape = {shape}, size = {size}, # zero = {cntzero}, sparsity = {sparsity}, # non-zero = {cntnzero}, density = {density}")
    return

# Find Pct%-close-to-0 values of a martrix and cast them to 0
def CastValueAroundZero(Mat: np.array, Pct: float) -> np.array:
    absMat = np.abs(Mat)
    eleCnt = Mat.size
    # Sort and cast
    sortIndx = np.unravel_index(np.argsort(absMat, axis=None), absMat.shape)
    sortMat = absMat[sortIndx]
    if Pct >= 1.0 or Pct <= 0.0:
        print("The input percentage should be between 0 and 1. The percentage is set to 0.5 by default.")
        Pct = 0.5
    thresIdx = int(eleCnt * Pct) 
    thresVal = sortMat[thresIdx]    
    Mat = np.where(absMat > thresVal, Mat, 0)
    return Mat