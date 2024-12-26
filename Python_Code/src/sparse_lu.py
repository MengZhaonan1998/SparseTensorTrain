import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import warnings

def sparse_outer_product_lu(A):
    """
    Compute the LU decomposition of a sparse matrix using outer product method.
    
    Parameters:
    -----------
    A : scipy.sparse matrix
        Input matrix in CSC format
        
    Returns:
    --------
    L : scipy.sparse.csc_matrix
        Lower triangular matrix with unit diagonal
    U : scipy.sparse.csc_matrix
        Upper triangular matrix
    """
    # Convert input to CSC format if not already
    if not isinstance(A, sparse.csc_matrix):
        A = csc_matrix(A)
    
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    # Initialize L and U as LIL matrices for efficient element access
    L = lil_matrix((n, n))
    U = lil_matrix((n, n))
    
    # Set unit diagonal for L
    L.setdiag(1)
    
    # Working copy of A
    A_work = A.tolil()
    
    for k in range(n):
        # Get pivot element
        pivot = A_work[k, k]
        if abs(pivot) < 1e-12:
            warnings.warn(f"Near-zero pivot encountered at position ({k},{k})")
            pivot = 1e-12
            
        # Fill kth row of U
        U[k, k:] = A_work[k, k:]
        
        # Compute multipliers and fill kth column of L
        if k < n-1:
            L[k+1:, k] = A_work[k+1:, k] / pivot
            
            # Outer product update - only on non-zero pattern
            # Get non-zero patterns
            row_pattern = U[k, k:].nonzero()[1]
            col_pattern = L[k+1:, k].nonzero()[0]
            
            if len(row_pattern) > 0 and len(col_pattern) > 0:
                for i in col_pattern:
                    for j in row_pattern:
                        A_work[i+k+1, j+k] -= L[i+k+1, k] * U[k, j+k]
    
    return L.tocsc(), U.tocsc()

def verify_lu(A, L, U, tol=1e-10):
    """
    Verify the LU decomposition by checking if A ≈ L*U
    
    Parameters:
    -----------
    A : scipy.sparse matrix
        Original matrix
    L : scipy.sparse matrix
        Lower triangular factor
    U : scipy.sparse matrix
        Upper triangular factor
    tol : float
        Tolerance for comparison
        
    Returns:
    --------
    bool : True if decomposition is valid
    """
    diff = abs((L * U - A).max())
    return diff < tol

# Example usage and testing
def create_test_matrix(n, density=0.1):
    """
    Create a random sparse test matrix
    """
    # Create random sparse matrix
    A = sparse.random(n, n, density=density, format='csc', random_state=42)
    # Make it diagonally dominant to ensure stability
    A.setdiag(np.abs(A).sum(axis=1).A1 + 1)
    return A

def run_example():
    # Create test matrix
    n = 5
    A = create_test_matrix(n, density=0.3)
    print("Original matrix pattern:")
    print(A.toarray())
    
    # Compute LU decomposition
    L, U = sparse_outer_product_lu(A)
    
    print("\nL factor:")
    print(L.toarray())
    print("\nU factor:")
    print(U.toarray())
    
    # Verify decomposition
    is_valid = verify_lu(A, L, U)
    print(f"\nDecomposition {'is' if is_valid else 'is not'} valid")
    
    # Print sparsity information
    print(f"\nSparsity patterns:")
    print(f"A: {A.nnz} non-zeros")
    print(f"L: {L.nnz} non-zeros")
    print(f"U: {U.nnz} non-zeros")

if __name__ == "__main__":
    run_example()