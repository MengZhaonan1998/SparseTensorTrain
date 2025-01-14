#include <cuda_runtime.h>
#include "new/sptensor.h"
#include "new/spmatrix.h"
#include "new/util.h"
#include "new/structures.h"

void dCOOMatrix_l1_hMemFree(COOMatrix_l2<double> hM)
{
    
    return;
}

void dCOOMatrix_l1_copyH2H(COOMatrix_l1<double>& hM_dest, const COOMatrix_l1<double>& hM_src)
{
    // Input validation
    if (hM_src.row_indices == nullptr || 
        hM_src.col_indices == nullptr || 
        hM_src.values == nullptr) {
        throw std::invalid_argument("Source matrix contains null pointers");
    }

    // Clean up any existing memory in destination
    delete[] hM_dest.row_indices;
    delete[] hM_dest.col_indices;
    delete[] hM_dest.values;

    // Copy matrix properties
    hM_dest.rows = hM_src.rows;
    hM_dest.cols = hM_src.cols;
    hM_dest.nnz = hM_src.nnz;

    // Allocate memory and copy data only if nnz > 0
    if (hM_src.nnz > 0) {
        try {
            hM_dest.row_indices = new size_t[hM_src.nnz];
            hM_dest.col_indices = new size_t[hM_src.nnz];
            hM_dest.values = new double[hM_src.nnz];
            std::memcpy(hM_dest.row_indices, hM_src.row_indices, hM_src.nnz * sizeof(size_t));
            std::memcpy(hM_dest.col_indices, hM_src.col_indices, hM_src.nnz * sizeof(size_t));
            std::memcpy(hM_dest.values, hM_src.values, hM_src.nnz * sizeof(double));
        } catch (const std::bad_alloc& e) {
            // Clean up if allocation fails
            delete[] hM_dest.row_indices;
            delete[] hM_dest.col_indices;
            delete[] hM_dest.values;
            hM_dest.row_indices = nullptr;
            hM_dest.col_indices = nullptr;
            hM_dest.values = nullptr;
            throw std::runtime_error("Memory allocation failed in matrix copy");
        }        
    } else {
        // Set pointers to nullptr if nnz = 0
        hM_dest.row_indices = nullptr;
        hM_dest.col_indices = nullptr;
        hM_dest.values = nullptr;
    }

    return;
}

decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU(COOMatrix_l2<double> const M_, double const cutoff, 
                        double const spthres, size_t const maxdim, bool const isFullReturn)
{
    // Dimension argument check
    assertm(maxdim > 0, "maxdim must be positive");

    // Initialize maximum truncation dimension k and permutations
    size_t Nr = M_.rows;
    size_t Nc = M_.cols;
    size_t k = std::min(Nr, Nc);
    size_t* rps = new size_t[Nr];
    size_t* cps = new size_t[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    COOMatrix_l2<double> M(M_); // Copy input M_ to a M
    double inf_error = 0.0;     // Inference error 
    size_t s = 0;               // Iteration number
    bool denseFlag = false;     // Dense/Sparse switch flag
    
    // Sparse-style computation 
    // A question: Do we want to sort COO every time?
    // One thing to verify: how much sparsity we lose during this outer-product iteration?
    while (s < k) {
        // Sparse -> Dense criteria
        //std::cout << "Density = " << double(M.nnz_count) / Nr / Nc << std::endl;
        double density = double(M.nnz_count) / Nr / Nc;
        if (density > spthres) {
            denseFlag = true;
            break;
        }
        
        // Partial M, Mabs = abs=(M[s:,s:]), max value of Mabs        
        double Mabs_max = 0.0;
        double Mdenom;
        size_t max_idx;
        size_t nnz = M.nnz_count;
        for (size_t i = 0; i < nnz; ++i) {
            if (M.row_indices[i] >= s && M.col_indices[i] >= s) {
                double Mabs = std::abs(M.values[i]);
                if (Mabs > Mabs_max) {
                    Mabs_max = Mabs;
                    Mdenom = M.values[i];
                    max_idx = i;
                }
            }                
        }
        
        // termination condition
        if (Mabs_max < cutoff) {
            inf_error = Mabs_max;
            break;
        }

        // piv, swap rows and columns  // BENCHMARK COO VS CSR (Later)
        size_t piv_r = M.row_indices[max_idx];
        size_t piv_c = M.col_indices[max_idx];
        for (size_t i = 0; i < nnz; ++i) {
            if (M.row_indices[i] == s)
                M.row_indices[i] = piv_r;
            else if (M.row_indices[i] == piv_r)
                M.row_indices[i] = s;
            if (M.col_indices[i] == s)
                M.col_indices[i] = piv_c;
            else if (M.col_indices[i] == piv_c)
                M.col_indices[i] = s;
        }

        // Sub-matrix update by outer-product
        if (s < k - 1) {
            for (size_t i = 0; i < nnz; ++i) {
                if (M.row_indices[i] == s && M.col_indices[i] > s) {
                    for (size_t j = 0; j < nnz; ++j) {                      
                        if (M.col_indices[j] == s && M.row_indices[j] > s) {
                            double outprod = M.values[j] * M.values[i] / Mdenom;
                            M.addUpdate(M.row_indices[j], M.col_indices[i], -1.0 * outprod);
                        }
                    }
                }        
            }
        }

        size_t temp;
        temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
        temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;
        s += 1;
    }

    // Dense-style computation 
    double* M_full;
    if (denseFlag) {
        M_full = M.todense();
        M.explicit_destroy();
        while (s < k) {
            // Partial M, Mabs = abs(M[s:,s:])    
            double Mabs_max = 0.0;
            size_t piv_r;
            size_t piv_c;
            for (size_t i = s; i < Nr; ++i)
                for (size_t j = s; j < Nc; ++j) {
                    double Mabs = std::abs(M_full[i * Nc + j]);
                    if (Mabs > Mabs_max) {
                        Mabs_max = Mabs;
                        piv_r = i;
                        piv_c = j;
                    }   
                }
            
            // termination condition
            if (Mabs_max < cutoff) {
                inf_error = Mabs_max;
                break;
            }

            // Row/Column swap
            cblas_dswap(Nc, M_full +  piv_r * Nc, 1, M_full + s * Nc, 1);
            cblas_dswap(Nr, M_full + piv_c, Nc, M_full + s, Nc);
            
            // Outer-product update 
            if (s < k - 1) {
                for (size_t i = s + 1; i < Nr; ++i) {
                    for (size_t j = s + 1; j < Nc; ++j) {
                        double outprod = M_full[i * Nc + s] * M_full[s * Nc + j] / M_full[s * Nc + s];
                        M_full[i * Nc + j] = M_full[i * Nc + j] - outprod;
                    }
                }
            }

            // Swap rps, cps
            size_t temp;
            temp = rps[s]; rps[s] = rps[piv_r]; rps[piv_r] = temp;
            temp = cps[s]; cps[s] = cps[piv_c]; cps[piv_c] = temp;

            s += 1;
        } 
    }

    size_t rank = s;  // Detected matrix rank    
    size_t output_rank = std::min(maxdim, rank);
    //util::PrintMatWindow(M_full, Nr, Nc, {0, Nr-1}, {0, Nc-1}); 

    // Result set
    decompRes::SparsePrrlduRes<double> resultSet;
    resultSet.rank = rank;  // Detected real rank
    resultSet.output_rank = output_rank;   // Output (truncated) rank
    resultSet.inf_error = inf_error;    // Inference error
    resultSet.isSparseRes = !denseFlag; // Dense or Sparse result
    resultSet.isFullReturn = isFullReturn; // Full or non-full return
    
    // Create inverse permutations
    resultSet.col_perm_inv = new size_t[Nc]{0};
    resultSet.row_perm_inv = new size_t[Nr]{0};
    for (size_t i = 0; i < Nr; ++i)
        resultSet.row_perm_inv[rps[i]] = i;
    for (size_t j = 0; j < Nc; ++j)
        resultSet.col_perm_inv[cps[j]] = j;
    delete[] rps;
    delete[] cps;

    if (denseFlag) {   
        if (isFullReturn) {
            // Memory allocation
            resultSet.d = new double[output_rank]{0.0};
            resultSet.dense_L = new double[Nr * output_rank]{0.0};
            resultSet.dense_U = new double[output_rank * Nc]{0.0};
            double* L = resultSet.dense_L;
            double* U = resultSet.dense_U;
            double* d = resultSet.d;
            
            // Diagonal entries
            for (size_t i = 0; i < output_rank; ++i)
                L[i * output_rank + i] = 1.0;
            for (size_t i = 0; i < output_rank; ++i)
                U[i * Nc + i] = 1.0;

            // Rank-revealing Guassian elimination
            for (size_t ss = 0; ss < output_rank; ++ss) {
                double P = M_full[ss * Nc + ss];
                d[ss] = P; 

                // Gaussian elimination
                if (ss < Nr - 1) {
                    // pivoted col
                    for (size_t i = ss + 1; i < Nr; ++i)
                        L[i * output_rank + ss] = M_full[i * Nc + ss] / P;
                }
                if (ss < Nc - 1) {
                    // pivoted row
                    for (size_t j = ss + 1; j < Nc; ++j)
                        U[ss * Nc + j] = M_full[ss * Nc + j] / P;
                }
            }
        } else {
            // TODO...
        }
        
        //std::cout << "D\n";
        //util::Print1DArray(resultSet.d, output_rank); 
        //std::cout << "L\n";
        //util::PrintMatWindow(resultSet.dense_L, Nr, output_rank, {0,Nr-1},{0,output_rank-1});
        //std::cout << "U\n";
        //util::PrintMatWindow(resultSet.dense_U, output_rank, Nc, {0,output_rank-1}, {0,Nc-1});
        delete[] M_full;
        return resultSet;
    } else {
        // Todo...
        if (isFullReturn) {

        } else {
            // TODO...
        }
        
        return resultSet;
    }
}

decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU(COOMatrix_l2<double> const M, double const cutoff, 
                        double const spthres, size_t const maxdim)
{   
    // Result set initialization
    decompRes::SparseInterpRes<double> idResult;

    // Partial rank-revealing LDU 
    // cutoff / spthres / maxdim are controlled by input arguments of interpolative function
    // isFullReturn for prrldu function is set to TRUE by default so far
    bool isFullReturn_prrldu = true;
    auto prrlduResult = dSparse_PartialRRLDU_CPU(M, cutoff, spthres, maxdim, isFullReturn_prrldu);
    
    // Rank detection
    idResult.rank = prrlduResult.rank;
    idResult.output_rank = prrlduResult.output_rank;
    
    size_t output_rank = prrlduResult.output_rank;
    size_t Nr = M.rows;
    size_t Nc = M.cols;

    // Get pivot columns (CPU part)
    idResult.pivot_cols = new size_t[Nc];
    for (size_t i = 0; i < Nc; ++i) {
        size_t idx;
        for (size_t j = 0; j < Nc; ++j) {
            if (prrlduResult.col_perm_inv[j] == i) {
                idx = j;
                break;
            }     
        }
        idResult.pivot_cols[i] = idx;
    }

    // Allocate memory for interpolative coefficients
    idResult.interp_coeff = new double[output_rank * (Nc - output_rank)]{0.0};
            
    // Interpolation coefficients
    if (prrlduResult.isSparseRes) {
        // Sparse U -> Sparse interpolation
        if (prrlduResult.isFullReturn) {
            // TODO...    
        } else {
            // If the results are returned in economic mode, we need an another implementation
            // TODO...
        }
    } else {
        // Dense U -> Dense interpolation
        if (prrlduResult.isFullReturn) {            
            double* U11 = new double[output_rank * output_rank]{0.0};
            double* b = new double[output_rank]{0.0};
            
            // Extract relevant submatrices
            for (size_t i = 0; i < output_rank; ++i)
                std::copy(prrlduResult.dense_U + i * Nc, prrlduResult.dense_U + i * Nc + output_rank, U11 + i * output_rank);

            // Compute the interpolative coefficients through solving upper triangular systems
            for (size_t i = output_rank; i < Nc; ++i) {
                // Right hand side b (one column of the U)
                for (size_t j = 0; j < output_rank; ++j)
                    b[j] = prrlduResult.dense_U[j * Nc + i];

                // Triangular solver (BLAS)        
                cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, output_rank, U11, output_rank, b, 1);

                // Copy the solution to iU11 columns
                for (size_t j = 0; j < output_rank; ++j) 
                    idResult.interp_coeff[j * (Nc - output_rank) + (i - output_rank)] = b[j];                
            }

            // Memory release
            delete[] b;
            delete[] U11;
        } else {
            // If the results are returned in economic mode, we need an another implementation
            // TODO...
        }
    }   

    // Memory release
    prrlduResult.freeSpLduRes();
    return idResult;
}

COOMatrix_l2<double> dcoeffZRecon(double* coeffMatrix, size_t* pivot_col, size_t rank, size_t col)
{
    COOMatrix_l2<double> Z(rank, col);
    
    // Identity part
    for (size_t i = 0; i < rank; ++i) {
        Z.add_element(i, pivot_col[i], 1.0);
    }   

    // Coefficient part
    for (size_t i = rank; i < col; ++i) {   
        for (size_t r = 0; r < rank; ++r) {
            Z.add_element(r, pivot_col[i], coeffMatrix[r * (col - rank) + (i - rank)]);
        }
    }

    return Z;
}

