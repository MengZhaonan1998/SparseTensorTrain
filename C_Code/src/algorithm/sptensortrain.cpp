#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/functions.h"
#include "new/structures.h"
#include "new/util.h"

// Define the template function TT_ID_sparse
template<typename T, size_t Order>
SparseTTRes TT_ID_sparse(const COOTensor<T, Order>& tensor, double const cutoff, 
                double const spthres, size_t const r_max, bool verbose) {
    // Initial settings
    auto shape = tensor.get_dimensions();  // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();                // Get the number of dimension d    
    size_t nbar = 1;                       // Get the total size: n1 * n2 * ... * nd 
    for (int i = 0; i < dim; ++i) nbar *= shape[i];

    // Reshape the input tensor to the 2D matrix
    size_t r = 1;  // Rank
    size_t row = nbar / r / shape[dim - 1];  // Initial matrix row
    size_t col = r * shape[dim - 1];         // Initial matrix column
    COOMatrix_l2<T> W = tensor.reshape2Mat(row, col);

    // Initialize a result list, start TT iteration
    SparseTTRes ResList; 
    ResList.InterG.resize(dim - 2);
    for (int i = dim - 1; i > 0; i--) {
        std::cout << "Tensor train iteration " << i << " starts..." <<  std::endl;
        
        // Reshape matrix (skip the first iteration)
        if (i != dim - 1) {            
            row = nbar / r / shape[i];  // Matrix row
            col = r * shape[i];         // Matrix column
            W.reshape(row, col);
        }

        // Sparse interpolative decomposition
        auto idResult = dSparse_Interpolative_CPU(W, cutoff, spthres, r_max);
        
        // There is no cutoff selection. Rank is revealed automatically by IDQR
        size_t ri = idResult.output_rank;
        
        auto Z = dcoeffZRecon(idResult.interp_coeff, idResult.pivot_cols, ri, col);
        //std::cout << "Factor Z nnz: " << Z.nnz_count << ", density: " << double(Z.nnz_count) / Z.cols / Z.rows << std::endl;

        // Print the low rank approximation error and sparse information
        if (verbose) {            
            auto C = W.subcol(idResult.pivot_cols, ri);
            //auto recon = C.multiply(Z);     // There are bugs in .multiply function?
            T max_error = 0.0;            
            for (size_t i = 0; i < W.rows; ++i)
                for (size_t j = 0; j < W.cols; ++j) {
                    T ele = 0.0; 
                    for (size_t k = 0; k < C.cols; ++k) {
                        ele += C.get(i, k) * Z.get(k, j);
                    }
                    T abs_error = std::abs(ele - W.get(i, j));
                    max_error = std::max(max_error, abs_error);
                }
            std::cout << "Interpolative reconstruction error: " << max_error << std::endl;   
        }

        // Form a new tensor-train factor
        if (i == dim - 1) {
            ResList.EndG.reset(Z.nnz_count, ri, shape[i]);
            ResList.EndG.nnz_count = Z.nnz_count;
            std::copy(Z.values, Z.values + Z.nnz_count, ResList.EndG.values);
            std::copy(Z.row_indices, Z.row_indices + Z.nnz_count, ResList.EndG.indices[0]);
            std::copy(Z.col_indices, Z.col_indices + Z.nnz_count, ResList.EndG.indices[1]);
        } else {
            ResList.InterG[i-1].reset(Z.nnz_count, ri, shape[i], r);
            ResList.InterG[i-1].nnz_count = Z.nnz_count;
            std::copy(Z.values, Z.values + Z.nnz_count, ResList.InterG[i-1].values);
            std::copy(Z.row_indices, Z.row_indices + Z.nnz_count, ResList.InterG[i-1].indices[0]);
            for (size_t n = 0; n < Z.nnz_count; ++n) {
                ResList.InterG[i-1].indices[1][n] = Z.col_indices[n] / r; 
                ResList.InterG[i-1].indices[2][n] = Z.col_indices[n] % r;
            }
        }
        
        // Form new W from the interpolative factor C
        nbar = nbar * ri / shape[i] / r;  // New total size of W
        r = ri;
        W = W.subcol(idResult.pivot_cols, ri);
    }

    // Append the last factor
    ResList.StartG.reset(W.nnz_count, W.rows, W.cols);
    ResList.StartG.nnz_count = W.nnz_count;
    std::copy(W.values, W.values + W.nnz_count, ResList.StartG.values);
    std::copy(W.row_indices, W.row_indices + W.nnz_count, ResList.StartG.indices[0]);
    std::copy(W.col_indices, W.col_indices + W.nnz_count, ResList.StartG.indices[1]);

    return ResList;
}

// Explicitly instantiate the specializations
template SparseTTRes TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, double, double, size_t, bool);
template SparseTTRes TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, double, double, size_t, bool);
template SparseTTRes TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, double, double, size_t, bool);
template SparseTTRes TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, double, double, size_t, bool);