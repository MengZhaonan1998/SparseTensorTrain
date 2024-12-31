#include <cuda_runtime.h>
#include "new/sptensor.h"
#include "new/spmatrix.h"

void dCOOMatrix_l1_hMemFree(COOMatrix_l1<double> hM)
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

void dSparse_PartialRRLDU_CPU(COOMatrix_l1<double> M_, double cutoff, size_t maxdim, size_t mindim)
{
    // Dimension argument check
    assertm(maxdim > 0, "maxdim must be positive");
    assertm(mindim > 0, "mindim must be positive");
    assertm(maxdim >= mindim, "maxdim must be larger than or equal to mindim");

    // Copy input M_ to a M
    COOMatrix_l1<double> M; 
    dCOOMatrix_l1_copyH2H(M, M_);

    // Initialize maximum truncation dimension k and permutations
    size_t Nr = M.rows;
    size_t Nc = M.cols;
    size_t k = std::min(Nr, Nc);
    size_t* rps = new size_t[Nr];
    size_t* cps = new size_t[Nc];
    std::iota(rps, rps + Nr, 0);
    std::iota(cps, cps + Nc, 0);

    // Find pivots
    double inf_error = 0.0;
    size_t s = 0;
    while (s < k) {
        // A question: Do we want to sort COO every time?
        
        // One thing to verify: how much sparsity we lose during this outer-product iteration?

        s += 1;
    }

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

