using LinearAlgebra
using Printf
using ITensorTCI: interpolative

function interpolative_nuclear(M::Matrix; cutoff=0.0, maxdim=min(size(M)))
  maxdim = min(maxdim, size(M)...)
  cols = Int[]
  K = M' * M
  m = size(K, 1)
  Kt = K
  error = 0.0
  for t in 1:maxdim
    Kt2 = Kt * Kt
    l = argmax(p -> (p ∈ cols ? -Inf : Kt2[p, p] / Kt[p, p]), 1:m)
    max_err2 = Kt2[l, l] / Kt[l, l]
    push!(cols, l)
    error = sqrt(abs(max_err2))
    (max_err2 < cutoff^2) && break
    Kt = K - K[:, cols] * (K[cols, cols] \ K[cols, :]) #Schur complement
  end
  C = M[:, cols]
  X = C \ M

  for w in 1:length(cols), r in 1:length(cols)
    X[r, cols[w]] = (r == w) ? 1.0 : 0.0
  end
  return C, X, cols, error
end

let
  M = zeros(5,6)
  M[1,2] = 1
  M[2,4] = 1
  cutoff = 1E-3
  maxdim = 2
  println("M = "); display(M); println()

  C, X, cols, error = interpolative_nuclear(M; cutoff, maxdim)

  println("C = "); display(C); println()
  println("X = "); display(X); println()
  @printf("Two-norm error = %.3E\n", norm(C*X - M, 2))
  @printf("∞-norm error = %.3E\n", norm(C*X - M, Inf))

  return
end

let
  A = [3 1; 8 2; 9 -5; -7 4]
  B = [4 6 2; 8 -1 -4]
  M = A * B
  cutoff = 1E-3
  maxdim = 2
  println("M = "); display(M); println()

  C, X, cols, error = interpolative_nuclear(M; cutoff, maxdim)

  println("C = "); display(C); println()
  println("X = "); display(X); println()
  @printf("Two-norm error = %.3E\n", norm(C*X - M, 2))
 @printf("∞-norm error = %.3E\n", norm(C*X - M, Inf))

  return
end

let
    n = 10
    m = 30
    k = 4
    eps = 1E-5
    cutoff = 1E-3
    maxdim = 7
   
    # Make an n×m matrix of approximate rank k
    #M = randn(n,m)
    M = randn(n,k)*randn(k,m) + eps*randn(n,m)
    
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
  
    C, Z, piv_cols, inf_err = interpolative(M; cutoff, maxdim)
  
    @show piv_cols
    println("C = "); display(C); println()
    println("Z = "); display(Z); println()
    @printf("Two-norm error = %.3E\n", norm(C*Z - M, 2))
    @printf("∞-norm error = %.3E\n", norm(C*Z - M, Inf))
  
    return
end