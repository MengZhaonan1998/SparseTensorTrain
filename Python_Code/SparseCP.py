import numpy as np
import tensorly as tl
import utils as ut

# For CP decomposition
from tensorly.decomposition import parafac

# Display
def PrintBunchInfo(Bunch):
    bunchName = [name for name, value in globals().items() if value is Bunch][0]
    dimension = Bunch['dims']    
    dimShape = np.shape(Bunch['tensor'])
    print("\nInformation of the tensorly bunch data loaded is given below:")
    print(f"Name: {bunchName}")
    print(f"Dimension: {dimension}")
    print(f"Shape: {dimShape}\n")
    return 

# Load some data built in tensorly
IL2 = tl.datasets.load_IL2data()
C19 = tl.datasets.load_covid19_serology()
KIN = tl.datasets.load_kinetic()
TensorBunchList = [C19, KIN] # there are some problems with IL2 dataset I think

for Data in TensorBunchList:
    # Load and display data
    PrintBunchInfo(Data)
    X = np.array(Data["tensor"])

    #Y = tl.tensor(np.arange(13*4*12*8).reshape((13, 4, 12, 8)), dtype=tl.float32)
    # CP decomposition of the tensor X
    weights, factors = parafac(X, rank=5)  # higher rank means .. higher precision of decomposition?

    # For hard/normalized sparsity, what is a good choice for the threshold? TO BE DISCUSSED...
    iter = 0
    sparse_factors = []
    for fMat in factors:
        cntEle = fMat.size
        fMat_cast = ut.CastValueAroundZero(fMat, 0.2)
        cntZero = (fMat_cast == 0.0).sum()
        print(f"factor {iter}: size = {cntEle}, #zero elements = {cntZero}, sparse density = {cntZero/cntEle}")
        sparse_factors.append(fMat_cast)
        iter += 1
    
    # Reconstruction
    reconT_nosparse = tl.cp_to_tensor((weights, factors))
    reconT_sparse = tl.cp_to_tensor((weights, sparse_factors)) 
    normError_nosparse = ut.NormError(X, reconT_nosparse, 2)
    normError_sparse = ut.NormError(X, reconT_sparse, 2)
    print(f"normError_nosparse / normError_sparse = {normError_nosparse} / {normError_sparse}")

    
        
    



