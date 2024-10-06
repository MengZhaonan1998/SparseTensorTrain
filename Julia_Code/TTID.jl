using LinearAlgebra
using Printf
using ITensorTCI: interpolative
using ITensors
using NDTensors

function TTSVD(T::ITensor, r_max::Int64, eps::Float64)
  idx = size(T)    # shape(index) of the input ITensor T
  dim = order(T)   # dimension number
  delta = (eps / sqrt(dim - 1)) * norm(T)  # Truncation parameter
  nbar = 1         # total size of T = i_x * i_y * i_z...
  for i in 1:dim
    nbar *= idx[i]
  end
  r = 1         # rank
  W = array(T)  # copy tensor T -> W
  factors = []
  for i in dim:-1:2
    reshapeR = Int(nbar / r / idx[i])
    reshapeC = Int(r * idx[i])
    W = reshape(W, (reshapeR, reshapeC)) # reshape   
    #@show W
    U, S, V = svd(W)   # Singular value decomposition of W
    Vh = transpose(V)  # Transpose V -> V^T
    # Compute rank r
    s = 0
    j = length(S)
    while s <= delta * delta
      s += S[j] * S[j]
      j -= 1
    end
    j += 1
    ri = min(j, r_max)
    vv = Vh[1:ri, :]
    Ti = reshape(vv, (ri, idx[i], r))
    nbar = Int(nbar * ri / idx[i] / r)
    r = ri
    W = U[:, 1:ri] * Diagonal(S[1:ri])
    #@show W
    pushfirst!(factors, Ti)
  end
  Ti = reshape(W, (1, idx[1], r))
  pushfirst!(factors, Ti)
  return factors
end

function TTID(T::ITensor, r_max::Int64, eps::Float64)
  idx = size(T)    # shape(index) of the input ITensor T
  dim = order(T)   # dimension number
  delta = (eps / sqrt(dim - 1)) * norm(T)  # Truncation parameter
  nbar = 1         # total size of T = i_x * i_y * i_z...
  for i in 1:dim
    nbar *= idx[i]
  end
  r = 1         # rank
  W = array(T)  # copy tensor T -> W
  factors = []
  for i in dim:-1:2
    reshapeR = Int(nbar / r / idx[i])
    reshapeC = Int(r * idx[i])
    W = reshape(W, (reshapeR, reshapeC)) # reshape   
    #@show W
    
    U, S, V = svd(W)   # Singular value decomposition of W
    

    cutoff = 1E-3
    C, Z, piv_cols, inf_err = interpolative(Float64.(W); cutoff)
    @printf("Two-norm error = %.3E\n", norm(C*Z - W, 2))
    @printf("∞-norm error = %.3E\n", norm(C*Z - W, Inf))
    shapeC = size(C)
    shapeZ = size(Z)
    
    ri = shapeC[2]

    Vh = Z  # Transpose V -> V^T
    # Compute rank r
    #s = 0
    #j = length(S)
    #while s <= delta * delta
    #  s += S[j] * S[j]
    #  j -= 1
    #end
    #j += 1
    #ri = min(j, r_max)
    
    
    vv = Vh[1:ri, :]
    Ti = reshape(vv, (ri, idx[i], r))
    nbar = Int(nbar * ri / idx[i] / r)
    r = ri
    W = C[:, 1:ri] #U[:, 1:ri] * Diagonal(S[1:ri])
    #@show W
    pushfirst!(factors, Ti)
  end
  Ti = reshape(W, (1, idx[1], r))
  pushfirst!(factors, Ti)
  return factors
end

function TTContraction(factors, Indices)
  iTlist = []
  facStart = factors[1]
  idxres = size(facStart)
  P = Indices[1]
  B = Index(idxres[3], "B")
  iTStart = ITensor(facStart, P, B)
  push!(iTlist, iTStart)

  n = length(factors)
  for i in 2:n-1
    factor = factors[i]
    idxres = size(factor)
    L = Indices[i]
    Q = Index(idxres[3], "Q")
    iTmid = ITensor(factor, B, L, Q)
    push!(iTlist, iTmid)
    B=Q
  end

  facEnd = factors[end]
  idxres = size(facEnd)
  P = Indices[end]
  iTEnd = ITensor(facEnd, B, P)
  push!(iTlist, iTEnd)

  recT = iTlist[1]
  for i in 2:n
    recT = recT * iTlist[i]
  end
  return recT  

end

function ErrorEval(T, recT)
  return norm(T - recT) / norm(T)
end

function TensorSparsityStat(T)
  idx = size(T)    # shape(index) of the input ITensor T
  dim = length(idx)   # dimension number
  arrayT = array(T)
  absT = broadcast(abs, arrayT)
  zeroList = findall(<(1E-14), absT)
  cntZero = size(zeroList)[1]
  nbar = 1         # total size of T = i_x * i_y * i_z...
  for i in 1:dim
    nbar *= idx[i]
  end
  cntNz = nbar - cntZero
  density = cntNz / nbar
  @printf("Density = %.3E\n", density)
  @printf("Number of non-zeros = %d\n", cntNz)
end


let
  # Try out TTID ?/
  I = Index(3, "index_i")
  J = Index(4, "index_j")
  K = Index(5, "index_k")
  M = Index(2, "index_m")
  N = Index(3, "index_n")
  Indices = (I, J, K, M, N)
  println(Indices)

  T = random_itensor(I, J, K, M, N)  
  # UGLY BLOCK
  sparsity = 0.80
  for i in 1:dim(I)
    for j in 1:dim(J)
      for k in 1:dim(K)
        for m in 1:dim(M)
          for n in 1:dim(N)
            if rand() < sparsity
              T[i,j,k,m,n] = 0.0
            end
          end
        end
      end
    end
  end

  TensorSparsityStat(T)

  r_max = 20
  eps = 1E-8
  factors = TTID(T, r_max, eps)  
  recT = TTContraction(factors, Indices)

  err = ErrorEval(T, recT)
  @show err

  # Sparsity statistics
  for i in 1:length(Indices)
    TensorSparsityStat(factors[i])
  end
  
end


let 
  # Try out TTSVD
  I = Index(3, "index_i")
  J = Index(4, "index_j")
  K = Index(5, "index_k")
  M = Index(2, "index_m")
  N = Index(3, "index_n")
  Indices = (I, J, K, M, N)
  println(Indices)

  
  T = ITensor(I, J, K, M, N)
  for i in 1:dim(I)
    for j in 1:dim(J)
      for k in 1:dim(K)
        for m in 1:dim(M)
          for n in 1:dim(N)
            T[i,j,k,m,n] = (i-1) - (j-1) + (k-1) - (m-1) + (n-1)                    
          end
        end
      end
    end
  end
  #@show T

  r_max = 5
  eps = 1E-4
  factors = TTSVD(T, r_max, eps)
  @show factors
  
  recT = TTContraction(factors, Indices)
  @show recT


  err = ErrorEval(T, recT)
  @show err
  
  
  #=
  fac1 = factors[1]
  idxres = size(fac1)
  P = Index(idxres[2], "P")
  B = Index(idxres[3], "B")
  iT1 = ITensor(fac1, P, B)


  

  fac2 = factors[2]
  idxres = size(fac2)
  L = Index(idxres[2], "L")
  Q = Index(idxres[3], "Q")
  iT2 = ITensor(fac2, B, L, Q)
  B=Q

  fac3 = factors[3]
  idxres = size(fac3)
  P = Index(idxres[2], "P")
  iT3 = ITensor(fac3, B, P)

  iT = iT1 * iT2 * iT3
  @show iT

  for i = 1:order(T)-1
    factorL = factors[i]
    factorR = factors[i+1]
    idxL = size(factorL)
    idxR = size(factorR)
    I = Index(idxL[1], "index_i")
    J = Index(idxL[2], "index_j")
    K = Index(idxL[3], "index_k")
    M = Index(idxR[2], "index_m")
    N = Index(idxR[3], "index_n")
    iTL = ITensor(factorL, I, J, K)
    ITR = ITensor(factorR, K, M, N)
    
  end
  =#

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
    n = 300
    m = 3
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



