import numpy as np
import sparse as sp
import scipy as sc

import tensorly as tl
from tensorly.contrib.decomposition import tensor_train_cross
from tensorly.decomposition import tensor_train

from tensortrain_cross import sparse_ttcross
from tensortrain_svd import TT_SVD
from utils import TensorSparseStat
from utils import readfrostt

class chicago_crime_comm_dataConfig:
    dataPath = "/home/mengzn/Desktop/TensorData/chicago-crime-comm.tns"    
    order = 4
    #dimensions = (6186,24,77,32)
    dimensions = (40,20,30,20)

class uber_pickup_dataConfig:
    dataPath = "/home/mengzn/Desktop/TensorData/uber.tns"
    order = 4
    #dimensions = (183, 24, 1140, 1717)
    dimensions = (20, 10, 30, 40)

SpT_ccc = readfrostt(chicago_crime_comm_dataConfig.dataPath, chicago_crime_comm_dataConfig.dimensions)
#SpT_uber = readfrostt(uber_pickup_dataConfig.dataPath, uber_pickup_dataConfig.dimensions)

def UnitTest1():
    SpT = SpT_ccc 
    #SpT = SpT_uber
    
    print("Unit test 1 starts!")
    shape = SpT.shape
    dim = SpT.ndim
    print(f"The input sparse tensor: shape = {shape}, nnz = {SpT.nnz}, density = {SpT.density}")    
    
    SpTd = SpT.todense() # need to eliminate this some day
    maxdim = 39

    cutoff = 1e-5    
    factors = TT_SVD(SpTd, maxdim, cutoff)
    reconT = tl.tt_to_tensor(factors)
    print(f"The TTSVD reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(factors)
    
    rank = [1]
    for i in range(len(factors)-1):
        shape = factors[i+1].shape
        rank.append(shape[0])
        pass
    rank.append(1)
    print(f"The optimal rank is {rank}")
    
    spFactors = tensor_train_cross(SpTd, rank, cutoff)
    reconT = tl.tt_to_tensor(spFactors)
    print(f"The TTCross reconstruction error is {tl.norm(SpTd-reconT, 2)/tl.norm(SpTd, 2)}")
    TensorSparseStat(spFactors)
    
    print("Unit test 1 ends!")
    return

UnitTest1()
