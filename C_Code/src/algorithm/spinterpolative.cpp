#include <cuda_runtime.h>
#include "new/sptensor.h"
#include "new/spmatrix.h"

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

void dSparse_econPRRLDU_CPU()
{

}


void dSparse_PartialRRLDU_CPU(COOMatrix_l2<double> M_, double cutoff, size_t maxdim, size_t mindim)
{
    // Dimension argument check
    assertm(maxdim > 0, "maxdim must be positive");
    assertm(mindim > 0, "mindim must be positive");
    assertm(maxdim >= mindim, "maxdim must be larger than or equal to mindim");

    // Initialize maximum truncation dimension k and permutations
    size_t Nr = M_.rows;
    size_t Nc = M_.cols;
    size_t capacity = M_.capacity;
    size_t k = std::min(Nr, Nc);
    size_t* rps = new size_t[Nr];
    size_t* cps = new size_t[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    // Copy input M_ to a M
    COOMatrix_l2<double> M(M_); 

    double inf_error = 0.0;
    size_t s = 0;
    
    // Sparse-style computation 
    // A question: Do we want to sort COO every time?
    // One thing to verify: how much sparsity we lose during this outer-product iteration?
    while (s < k) {
        // Partial M, Mabs = abs=(M[s:,s:]), max value of Mabs
        double Mabs_max = 0.0;
        double Mdenom;
        size_t max_idx;
        size_t nnz = M.nnz_count;
        for (size_t i = 0; i < nnz; ++i) {
            if (M.row_indices[i] < (Nr - s) && M.col_indices[i] < (Nc - s)) {
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

        // piv, swap rows and columns
        size_t piv_r = M.row_indices[max_idx];
        size_t piv_c = M.col_indices[max_idx];
        for (size_t i = 0; i < nnz; ++i) {
            if (M.row_indices[i] == s)
                M.row_indices[i] = piv_r;
            if (M.row_indices[i] == piv_r)
                M.row_indices[i] = s;
            if (M.col_indices[i] == s)
                M.col_indices[i] = piv_c;
            if (M.col_indices[i] == piv_c)
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
    //while (s < k) {


    //    s += 1;
    //}
    
    return;
}

void dSparse_trsv()
{
    return;
}

void dSparse_Interpolative()
{

    return;
}

