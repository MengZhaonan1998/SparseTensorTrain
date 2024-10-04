using LinearAlgebra: norm
using Printf
using ITensorTCI: interpolative
using ITensors
using NDTensors

function TTID(T::ITensor, eps::Float64)

end

function TTSVD(T::ITensor, r_max::Int64, eps::Float64)
  idx = size(T)    # shape(index) of the input ITensor T
  dim = order(T)   # dimension number
  nbar = 1         # total size of T = i_x * i_y * i_z...
  for i in 1:dim
    nbar *= idx[i]
  end

  W = T  # copy tensor T -> W
  r = 1  # rank
  delta = (eps / sqrt(dim - 1)) * norm(T)    
    
  for i in dim:-1:2
      
       
  end
end

let 
  # Try out TTSVD
  I = Index(3, "index_i")
  J = Index(4, "index_j")
  K = Index(5, "index_k")
  T = ITensor(I, J, K)
  for i in 1:dim(I)
      for j in 1:dim(J)
          for k in 1:dim(K)
              T[i,j,k] = i+j+k-1-1-1              
          end
      end
  end
  @show T
  
  r_max = 5
  eps = 1E-4
  TTSVD(T, r_max, eps)

end

let 
  # Try out ITensor
  i = Index(2, "index_i")
  j = Index(4, "index_j")
  k = Index(3, "index_k") 
  T = random_itensor(i, j, k)
  @show typeof(T)
  @show T
    
  A = array(T)
  R = reshape(A, (2*4,3))
  @show R

  i = Index(2*4, "index_i")
  j = Index(3, "index_j")
  T = ITensor(R, i, j)
  @show T
end

let
    n = 3
    m = 20
    k = 4
    eps = 1E-5
    cutoff = 1E-3
   
    # Make an n×m matrix of approximate rank k
    M = randn(n,k)*randn(k,m) + eps*randn(n,m)
    M = randn(n,m) 
    
    # added by Meng 09/27/2024
    sparsity = 0.93
    for i in 1:n
      for j in 1:m
          if rand() < sparsity
            M[i,j] = 0
          end
      end
    end
  
    println("M = "); display(M); println()
  
    C, Z, piv_cols, inf_err = interpolative(M; cutoff)
  
    @show piv_cols
    println("C = "); display(C); println()
    println("Z = "); display(Z); println()
    @printf("Two-norm error = %.3E\n", norm(C*Z - M, 2))
    @printf("∞-norm error = %.3E\n", norm(C*Z - M, Inf))
  
    return
end



