/*
#include "header.h"

void synthetic_test_1()
{   std::cout << "Synthetic test 1 (dense TT, dense SVD) starts." << "\n";
    auto synTensor = util::SyntheticTenGen<double>({4,5,7,5},{2,3,4});
    auto ttList = TT_SVD_dense(synTensor, 5, 1E-10);
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(synTensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
    return;
}



int main(int argc, char** argv) 
{
    std::cout << "Synthetic test starts!" << std::endl;
    synthetic_test_2();
    std::cout << "Synthetic test ends!" << std::endl;
    return 0; 
}
*/

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <lapacke.h>

struct PRRLDUResult {
    double* L;
    double* d;
    double* U;
    int* row_perm;
    int* col_perm;
    double inf_error;
    int rank;
};

void find_max_abs(const double* M, int rows, int cols, int start_row, int start_col,
                  double& max_val, int& max_row, int& max_col) {
    max_val = 0.0;
    max_row = start_row;
    max_col = start_col;
    
    for (int i = start_row; i < rows; ++i) {
        for (int j = start_col; j < cols; ++j) {
            double abs_val = std::abs(M[j * rows + i]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = i;
                max_col = j;
            }
        }
    }
}

PRRLDUResult* prrldu(const double* M_in, int rows, int cols, double cutoff, int maxdim, int mindim = 1) {
    assert(maxdim > 0);
    assert(mindim > 0);
    mindim = std::min(maxdim, mindim);
    
    // Allocate result structure
    PRRLDUResult* result = new PRRLDUResult;
    int k = std::min(rows, cols);
    result->L = new double[rows * k]();  // initialize to zero
    result->d = new double[k]();
    result->U = new double[k * cols]();
    result->row_perm = new int[rows];
    result->col_perm = new int[cols];
    result->inf_error = 0.0;
    result->rank = 0;
    
    // Copy input matrix
    double* M = new double[rows * cols];
    std::memcpy(M, M_in, rows * cols * sizeof(double));
    
    // Initialize permutations
    for (int i = 0; i < rows; ++i) result->row_perm[i] = i;
    for (int i = 0; i < cols; ++i) result->col_perm[i] = i;
    
    // Initialize L and U as identity matrices
    for (int i = 0; i < k; ++i) {
        result->L[i * rows + i] = 1.0;
        result->U[i * cols + i] = 1.0;
    }
    
    // Main decomposition loop
    for (int s = 0; s < k && s < maxdim; ++s) {
        double max_val;
        int max_row, max_col;
        find_max_abs(M, rows-s, cols-s, s, s, max_val, max_row, max_col);
        
        if (max_val < cutoff) {
            result->inf_error = max_val;
            break;
        }
        
        // Adjust indices
        max_row += s;
        max_col += s;
        
        // Swap rows and columns
        if (max_row != s) {
            cblas_dswap(cols-s, &M[s*cols+s], rows, &M[max_row*cols+s], rows);
            std::swap(result->row_perm[s], result->row_perm[max_row]);
        }
        if (max_col != s) {
            cblas_dswap(rows-s, &M[s*cols+s], 1, &M[max_col*cols+s], 1);
            std::swap(result->col_perm[s], result->col_perm[max_col]);
        }
        
        double pivot = M[s*cols + s];
        result->d[s] = pivot;
        
        if (result->rank < mindim) {
            // proceed
        } else if (pivot == 0.0 || (std::abs(pivot) < cutoff && result->rank + 1 > mindim)) {
            break;
        }
        
        if (pivot == 0.0) pivot = 1.0;
        result->rank++;
        
        // Update L
        if (s < rows - 1) {
            for (int i = s + 1; i < rows; ++i) {
                result->L[s*rows + i] = M[s*cols + i] / pivot;
            }
        }
        
        // Update U
        if (s < cols - 1) {
            for (int j = s + 1; j < cols; ++j) {
                result->U[s*cols + j] = M[j*cols + s] / pivot;
            }
        }
        
        // Schur complement update
        if (s < k - 1) {
            for (int i = s + 1; i < rows; ++i) {
                for (int j = s + 1; j < cols; ++j) {
                    M[j*cols + i] -= M[s*cols + i] * M[j*cols + s] / pivot;
                }
            }
        }
    }
    
    delete[] M;
    return result;
}

struct InterpolativeResult {
    double* C;  // Selected columns matrix
    double* Z;  // Interpolation matrix
    int* pivot_columns;
    double inf_error;
    int rows;
    int cols;
    int rank;
};

// Helper function to solve linear system AX = B using LAPACKE
// Returns newly allocated array that caller must delete
double* solve_linear_system(const double* A, const double* B, int n, int nrhs) {
    // Create copies since LAPACKE_dgesv modifies inputs
    double* A_copy = new double[n * n];
    double* X = new double[n * nrhs];
    int* ipiv = new int[n];
    
    std::memcpy(A_copy, A, n * n * sizeof(double));
    std::memcpy(X, B, n * nrhs * sizeof(double));
    
    // Solve the system using LAPACKE_dgesv
    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A_copy, n, 
                            ipiv, X, n);
    
    // Cleanup
    delete[] A_copy;
    delete[] ipiv;
    
    if (info != 0) {
        delete[] X;
        throw std::runtime_error("Error in linear system solve");
    }
    
    return X;
}

// Helper function to create identity matrix
// Returns newly allocated array that caller must delete
double* create_identity(int n) {
    double* eye = new double[n * n]();  // Initialize to zero
    for (int i = 0; i < n; ++i) {
        eye[i * n + i] = 1.0;
    }
    return eye;
}

// Matrix multiplication helper: C = A * B
// Returns newly allocated array that caller must delete
double* matrix_multiply(const double* A, const double* B, int m, int k, int n) {
    double* C = new double[m * n]();  // Initialize to zero
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, m, B, k, 0.0, C, m);
    return C;
}

InterpolativeResult* interpolative_prrldu(const double* M, int rows, int cols,
                                        double cutoff = 0.0,
                                        int maxdim = std::numeric_limits<int>::max(),
                                        int mindim = 1) {
    // Compute PRRLDU decomposition
    PRRLDUResult* prrldu_result = prrldu(M, rows, cols, cutoff, maxdim, mindim);
    int k = prrldu_result->rank;
    
    // Create result structure
    InterpolativeResult* result = new InterpolativeResult;
    result->inf_error = prrldu_result->inf_error;
    result->rows = rows;
    result->cols = cols;
    result->rank = k;
    
    // Extract U11 (k x k upper left submatrix of U)
    double* U11 = new double[k * k];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            U11[j * k + i] = prrldu_result->U[j * cols + i];
        }
    }
    
    // Create identity matrix for solving system
    double* eye = create_identity(k);
    
    // Solve U11 * X = I to get inverse of U11
    double* iU11 = solve_linear_system(U11, eye, k, k);
    delete[] eye;
    delete[] U11;
    
    // Compute interpolation matrix ZjJ = iU11 @ U
    double* U_full = new double[k * cols];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < cols; ++j) {
            U_full[j * k + i] = prrldu_result->U[j * cols + i];
        }
    }
    double* ZjJ = matrix_multiply(iU11, U_full, k, k, cols);
    delete[] U_full;
    delete[] iU11;
    
    // Compute selected columns CIj = L @ diag(d) @ U11
    double* D = new double[k * k]();  // Initialize to zero
    for (int i = 0; i < k; ++i) {
        D[i * k + i] = prrldu_result->d[i];
    }
    
    double* L_full = new double[rows * k];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            L_full[j * rows + i] = prrldu_result->L[j * rows + i];
        }
    }
    
    // Extract U11 again for this computation
    U11 = new double[k * k];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            U11[j * k + i] = prrldu_result->U[j * cols + i];
        }
    }
    
    double* temp = matrix_multiply(D, U11, k, k, k);
    delete[] D;
    delete[] U11;
    
    double* CIj = matrix_multiply(L_full, temp, rows, k, k);
    delete[] L_full;
    delete[] temp;
    
    // Allocate and apply permutations to get final C and Z matrices
    result->C = new double[rows * k];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            result->C[j * rows + prrldu_result->row_perm[i]] = CIj[j * rows + i];
        }
    }
    delete[] CIj;
    
    result->Z = new double[k * cols];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < cols; ++j) {
            result->Z[prrldu_result->col_perm[j] * k + i] = ZjJ[j * k + i];
        }
    }
    delete[] ZjJ;
    
    // Get pivot columns
    result->pivot_columns = new int[k];
    for (int i = 0; i < k; ++i) {
        auto it = std::find(prrldu_result->col_perm, prrldu_result->col_perm + cols, i);
        result->pivot_columns[i] = std::distance(prrldu_result->col_perm, it);
    }
    
    // Clean up PRRLDU result
    delete[] prrldu_result->L;
    delete[] prrldu_result->d;
    delete[] prrldu_result->U;
    delete[] prrldu_result->row_perm;
    delete[] prrldu_result->col_perm;
    delete prrldu_result;
    
    return result;
}

// Helper function to clean up InterpolativeResult
void delete_interpolative_result(InterpolativeResult* result) {
    if (result) {
        delete[] result->C;
        delete[] result->Z;
        delete[] result->pivot_columns;
        delete result;
    }
}


#include <iostream>
#include <iomanip>

// Function to print a matrix
void print_matrix(const char* name, const double* M, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << M[j * rows + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Function to print an integer array
void print_array(const char* name, const int* arr, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n\n";
}


// Example usage
int main() {
    // Example matrix (3x4)
    const int rows = 3;
    const int cols = 4;
    double M[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    
    // Print original matrix
    std::cout << "Original matrix:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << M[j*rows + i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compute PRRLDU decomposition
    PRRLDUResult* result = prrldu(M, rows, cols, 1e-5, 3);
    
    // Print results
    std::cout << "\nRank: " << result->rank << std::endl;
    std::cout << "Inf error: " << result->inf_error << std::endl;
    
    std::cout << "\nL matrix:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < result->rank; ++j) {
            std::cout << result->L[j*rows + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nd vector:" << std::endl;
    for (int i = 0; i < result->rank; ++i) {
        std::cout << result->d[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nU matrix:" << std::endl;
    for (int i = 0; i < result->rank; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << result->U[j*cols + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nRow permutation:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << result->row_perm[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nColumn permutation:" << std::endl;
    for (int i = 0; i < cols; ++i) {
        std::cout << result->col_perm[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] result->L;
    delete[] result->d;
    delete[] result->U;
    delete[] result->row_perm;
    delete[] result->col_perm;
    delete result;
    
    return 0;
}