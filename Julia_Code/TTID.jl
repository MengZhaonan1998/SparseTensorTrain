using LinearAlgebra: norm
using Printf
using ITensorTCI: interpolative
using ITensors

let
    n = 3
    m = 20
    k = 4
    eps = 1E-5
    cutoff = 1E-3
  
    #
    # Make an n×m matrix of approximate rank k
    #
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

function TTID(T)
        
end

