include("TTID.jl")
using ITensors
using NDTensors

function temp_read_data(filename, shape)
    # Read all lines from the file
    lines = readlines(filename)

    # Split each line by spaces and convert to numbers
    data = [parse.(Int, split(line)) for line in lines]

    A = zeros(shape)

    for i = 1:length(data) 
        piece = data[i]
        flag1 = piece[1] <= shape[1]
        flag2 = piece[2] <= shape[2]
        flag3 = piece[3] <= shape[3]
        flag4 = piece[4] <= shape[4]
        if flag1 == true && flag2 == true && flag3 == true && flag4 == true 
            A[piece[1], piece[2], piece[3], piece[4]] = piece[5]
        end
    end    
    
    # Transpose the data to separate columns
    #data = hcat(data...)'

    return A
end


Tensor = temp_read_data("/home/mengzn/Desktop/TensorData/cc_short.tns", (40,20,30,20))

TensorSparsityStat(Tensor)

order = size(Tensor)
I = Index(order[1], "index_i")
J = Index(order[2], "index_j")
K = Index(order[3], "index_k")
L = Index(order[4], "index_l")
Indices = (I, J, K, L)
Tit = ITensor(Tensor, I, J, K, L)

@show I
@show J
@show K
@show L
@show size(Tit)

r_max = 40
eps = 1E-4
idFactors = TTID(Tit, r_max, eps)
recTensor = TTContraction(idFactors, Indices)

err = norm(Tit - recTensor) / norm(Tit)
println("The reconstruction error of TTID:")
@show err

# Sparsity statistics
println("The sparsity info of output TT factors:")
for i in 1:length(Indices)
    TensorSparsityStat(idFactors[i])
end
