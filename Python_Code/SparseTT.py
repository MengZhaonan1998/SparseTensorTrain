import numpy as np
import tensorly as tl
import sparse as sp
import scipy as sc
import tensorly.contrib.sparse as sptl
import tensorly.tenalg.proximal as prox

def legacy():
    rstate = 10
    shape = [20, 50, 16]
    rank = [1, 20, 20, 1]
    factors = tl.random.random_tt(shape, rank, random_state=rstate)
    tensor = tl.tt_to_tensor(factors)

    decompRank = [1, 2, 2, 1]
    factors = tl.decomposition.tensor_train(tensor, decompRank)
    recTensor = tl.tt_to_tensor(factors)
    print(tl.norm(recTensor-tensor, 2) / tl.norm(tensor))

    #factors[2] = prox.proximal_operator(factors[2], l1_reg=[0.1,0.1,0.1])
    #recTensor = tl.tt_to_tensor(factors)
    #print(tl.norm(recTensor-tensor, 2) / tl.norm(tensor))

    #vecd = tl.tensor_to_vec(factors[2])
    #vecs = prox.proximal_operator(vecd, l2_reg=0.1)
    #print(tl.norm(vecs-vecd, 2) / tl.norm(vecd))

    decompRank = [1, 10, 10, 1]
    factors = tl.decomposition.tensor_train(tensor, decompRank)
    for i in range(len(factors)):
        factor = factors[i]
        spFactor = prox.proximal_operator(factor, l1_reg=0.1)
        factors[i] = spFactor
    recTensor = tl.tt_to_tensor(factors)
    print(tl.norm(recTensor-tensor, 2) / tl.norm(tensor))

    '''
    threshold = 0.3
    for i in range(len(factors)):
        sub = np.abs(np.array(factors[i]))
        factors[i] = np.where(sub < threshold, 0, sub)
        pass

    recTensor = tl.tt_to_tensor(factors)
    print(tl.norm(recTensor-tensor, 2) / tl.norm(tensor))
    '''

    shape = (5, 3, 4)
    rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    spX = sp.random(shape, density=0.25, random_state=1, data_rvs=rvs)
    spXt = sptl.tensor(spX, dtype='float')

def legacy2():
    shape = (5, 3, 4, 5)
    rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    spX = sp.random(shape, density=0.25, random_state=1, data_rvs=rvs)
    #spXt = sptl.tensor(spX, dtype='float')
    spXt = spX.todense()
    rank = (1, 50, 50, 50, 1)
    factors = tl.decomposition.tensor_train(spXt, rank)
    reconXt = tl.tt_to_tensor(factors)
    print(tl.norm(reconXt-spXt, 2) / tl.norm(spXt, 2))

    factors[1] = prox.proximal_operator(factors[1], l1_reg=0.5)
    reconXt = tl.tt_to_tensor(factors)
    print(tl.norm(reconXt-spXt, 2) / tl.norm(spXt, 2))

def legacy3():
    shape = (2, 2, 20)
    rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    X = sp.random(shape, density=1, random_state=1, data_rvs=rvs)
    X = tl.tensor(X.todense())
    rank = (1, 3, 3, 1)
    factors = tl.decomposition.tensor_train(X, rank)
    reconX = tl.tt_to_tensor(factors)
    print(tl.norm(reconX-X, 2) / tl.norm(X))
    
    #factors[2] = prox.proximal_operator(factors[2], normalized_sparsity=0.01)
    #reconX = tl.tt_to_tensor(factors)
    #print(tl.norm(reconX-X, 2) / tl.norm(X))
    
def UndDetSys():
    row = 5
    col = 10
    A = np.random.rand(row, col)
    b = np.random.rand(row, 1)
    subA = A[:, 0:row]
    x = np.linalg.solve(subA, b)
    ex = np.zeros((col, 1))
    ex[0:row] = x
    print(tl.norm(A@ex-b, 2))

#UndDetSys() # CHECKED. It works
'''
# TODO: 3D TT try to solve the under determined linear system: T*M=b -> M
# It does not work...
shape = (3, 4, 6, 5)
rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
X = sp.random(shape, density=1, random_state=1, data_rvs=rvs)
X = tl.tensor(X.todense())
rank = (1, 6, 20, 6, 1)
factors = tl.decomposition.tensor_train(X, rank)
reconX = tl.tt_to_tensor(factors)
print(tl.norm(reconX-X, 2) / tl.norm(X))
'''

def RandomSparseFactors(shape: list, rank: list, density: float) -> list[np.array]:
    if rank[0] != 1 or rank[-1] != 1:
        print("The input rank needs to be zero at the beginning and end.")
        return [np.array([None])]
    if len(shape) != len(rank)-1:
        print("The input shape does not match with the input rank.")
        return [np.array([None])]
    factorList = []
    for i in range(len(shape)):
        #rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(i+1))
        factor = sp.random((rank[i], shape[i], rank[i+1]), density=density, random_state=i)
        factorList.append(factor)
    return factorList

#UndDetSys() # CHECKED. It works
def legacy4():
    shape = [31, 25, 28]
    rank = [1, 10, 10, 1]
    factorList = RandomSparseFactors(shape, rank, 0.01)
    for i in range(len(factorList)):
        factorList[i] = factorList[i].todense() 
    xx = tl.tt_to_tensor(factorList)
    tt = tl.decomposition.tensor_train(xx, rank)

shape = (5, 6, 10, 7)
density = 1e0
rvs = lambda x: sc.stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
T_COO = sp.random(shape, density=density, random_state=2, data_rvs=rvs)
T_NP = T_COO.todense()

rank = (1, 4, 4, 4, 1)
factors_NP = tl.decomposition.tensor_train(T_NP, rank)
RecT_NP = tl.tt_to_tensor(factors_NP)
print(tl.norm(RecT_NP - T_NP, 2) / tl.norm(T_NP, 2))

pass
