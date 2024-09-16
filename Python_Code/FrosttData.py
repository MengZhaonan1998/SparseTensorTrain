import numpy as np
import tensorly as tl
import sparse as sp

def readfrostt(path: str, shape: tuple):
    with open(path, "r") as file:
        modeList = []
        for line in file:
            numbers = line.strip().split()
            numbers = [int(num) for num in numbers]
            modeList.append(numbers)
        modeList = np.array(modeList, dtype=np.int32)
        modeList = modeList.T
        data = modeList[-1] 
        coords = modeList[:-1]-1
        spt = sp.COO(coords, data, shape)
    return spt

class chicago_crime_comm_dataConfig:
    dataPath = "/media/mengzn/A48084C98084A400/TensorData/Sparse/chicago-crime-comm.tns"    
    order = 4
    dimensions = (6186,24,77,32)

class uber_pickup_dataConfig:
    dataPath = "/media/mengzn/A48084C98084A400/TensorData/Sparse/uber.tns"
    order = 4
    dimensions = (183, 24, 1140, 1717)

#sparseTensor = readfrostt(chicago_crime_comm_dataConfig.dataPath, chicago_crime_comm_dataConfig.dimensions)
sparseTensor = readfrostt(uber_pickup_dataConfig.dataPath, uber_pickup_dataConfig.dimensions)

pass 