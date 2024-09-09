# Some evaluation methods are put in utils.py
import numpy as np
import tensorly as tl

# Error
def NormError(origT: np.array, reconT: np.array, order: int = 2) -> float:
    diffT = origT - reconT
    norm1 = tl.norm(diffT, order)
    norm2 = tl.norm(origT, order)
    normError = norm1 / norm2
    return normError

# Find x%-close-to-0 values of a martrix and cast them to 0
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

# Density