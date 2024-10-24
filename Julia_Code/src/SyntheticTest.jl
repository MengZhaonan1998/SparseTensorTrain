include("TTID.jl")
using SparseArrays
using Random

function RandomSpT3Gen(order, density)
    # random sparse tensor in dense format
    nbar = 1   
    dim = length(order)      
    for i in 1:dim
        nbar *= order[i]
    end
    V = sprandn(nbar, density)
    V = Array(V)
    V = reshape(V, order)
    return V
end

function RandomTTGen(order, rank, density, seed)
    # random sparse tensor train 
    factors = []
    dim = length(order)
    Random.seed!(seed)
    for i in 1:dim
        factorOrder = (rank[i], order[i], rank[i+1])
        factor = RandomSpT3Gen(factorOrder, density[i])  
        push!(factors, factor)
    end
    return factors    
end

function testCase1()
    # Same configuration with Python unit test 1 of TTCross/TTSVD
    # To compare...
    println("Unit Test 1 starts!")        
    order = (50, 50, 50)

    # Read all lines from the file
    filename = "/home/mengzn/Desktop/TensorData/syn_order_50_50_50_synrank_1_30_30_1.tns"
    lines = readlines(filename)
    data = [parse.(Float32, split(line)) for line in lines]
    Tensor = zeros(order)
    dim = length(order)
    nnz = length(data)
    for i = 1:nnz 
        piece = data[i]
        Tensor[map(Int,piece[1])+1, map(Int,piece[2])+1, map(Int,piece[3])+1] = piece[4]
    end    
    TensorSparsityStat(Tensor)
  
    I = Index(order[1], "index_i")
    J = Index(order[2], "index_j")
    K = Index(order[3], "index_k")
    Indices = (I, J, K)

    r_max = 40
    eps = 1E-5
    idFactors = TTID_Array(Tensor, r_max, eps)
    recTensor = TTContraction(idFactors, Indices)
    err = ErrorEval(Tensor, recTensor)
    println("The reconstruction error of TTID:")
    @show err
  
    # Sparsity statistics
    println("The sparsity info of output TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(idFactors[i])
    end
    println("Unit Test 2 ends!")
end

function testCase2()
    # Same configuration with Python unit test 2 of TTCross/TTSVD
    # To compare...
    println("Unit Test 2 starts!")
    order = (50, 50, 50, 50)
     
    # Read all lines from the file
    filename = "/home/mengzn/Desktop/TensorData/syn_order_50_50_50_50_synrank_1_50_200_50_1.tns"
    lines = readlines(filename)
    data = [parse.(Float32, split(line)) for line in lines]
    Tensor = zeros(order)
    dim = length(order)
    nnz = length(data)
    for i = 1:nnz 
        piece = data[i]
        Tensor[map(Int,piece[1])+1, map(Int,piece[2])+1, map(Int,piece[3])+1, map(Int,piece[4])+1] = piece[5]
    end    
    TensorSparsityStat(Tensor)

    I = Index(order[1], "index_i")
    J = Index(order[2], "index_j")
    K = Index(order[3], "index_k")
    L = Index(order[4], "index_l")
    Indices = (I, J, K, L)
    
    r_max = 200
    eps = 1E-5
    idFactors = TTID_Array(Tensor, r_max, eps)
    recTensor = TTContraction(idFactors, Indices)
    err = ErrorEval(Tensor, recTensor)
    println("The reconstruction error of TTID:")
    @show err

    # Sparsity statistics
    println("The sparsity info of output TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(idFactors[i])
    end
    println("Unit Test 2 ends!")
    return
end

function testCase3()
    println("Unit Test 2 starts!")
    order = [3, 6, 2, 4, 5]    
    rank = [1, 5, 9, 7, 8, 1]
    density = [0.2, 0.1, 0.2, 0.1, 0.2] 
    ttFactors = RandomTTGen(order, rank, density, 123)
    I = Index(order[1], "index_i")
    J = Index(order[2], "index_j")
    K = Index(order[3], "index_k")
    L = Index(order[4], "index_l")
    M = Index(order[5], "index_m")
    Indices = (I, J, K, L, M)
    println("The sparsity info of input TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(ttFactors[i])
    end
    println("The sparsity info of the synthetic input tensor:")
    Tensor = TTContraction(ttFactors, Indices)
    TensorSparsityStat(Tensor)

    r_max = 6
    eps = 1E-4
    idFactors = TTID(Tensor, r_max, eps)
    recTensor = TTContraction(idFactors, Indices)
    err = ErrorEval(Tensor, recTensor)
    println("The reconstruction error of TTID:")
    @show err

    # Sparsity statistics
    println("The sparsity info of output TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(idFactors[i])
    end
    println("Unit Test 2 ends")
end

function testCase4()
    println("Unit Test 3 starts!")
    order = [10, 8, 15, 7]    
    rank = [1, 10, 9, 7, 1]
    density = [0.1, 0.1, 0.1, 0.1] 
    ttFactors = RandomTTGen(order, rank, density, 123)
    I = Index(order[1], "index_i")
    J = Index(order[2], "index_j")
    K = Index(order[3], "index_k")
    L = Index(order[4], "index_l")
    Indices = (I, J, K, L)
    println("The sparsity info of input TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(ttFactors[i])
    end
    println("The sparsity info of the synthetic input tensor:")
    Tensor = TTContraction(ttFactors, Indices)
    TensorSparsityStat(Tensor)
  
    r_max = 10
    eps = 1E-4
    idFactors = TTID(Tensor, r_max, eps)
    recTensor = TTContraction(idFactors, Indices)
    err = ErrorEval(Tensor, recTensor)
    println("The reconstruction error of TTID:")
    @show err
  
    # Sparsity statistics
    println("The sparsity info of output TT factors:")
    for i in 1:length(Indices)
        TensorSparsityStat(idFactors[i])
    end
    println("Unit Test 3 ends")
end



let 
    println("Unit test starts!")
    #testCase1()
    testCase2()
    #testCase3()
    #testCase4()



    println("Unit test ends!")
end
