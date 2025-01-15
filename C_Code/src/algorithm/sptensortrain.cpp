#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/functions.h"

// Define the template function TT_ID_sparse
template<typename T, size_t Order>
void TT_ID_sparse(const COOTensor<T, Order>& tensor, double const cutoff, 
                double const spthres, size_t const r_max, bool verbose) {
    // Initial settings
    auto shape = tensor.get_dimensions();  // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();                // Get the number of dimension d    
    size_t nbar = 1;                       // Get the total size: n1 * n2 * ... * nd 
    for (int i = 0; i < dim; ++i)
        nbar *= shape[i];

    // Reshape the input tensor to the 2D matrix
    size_t r = 1;  // Rank
    size_t row = nbar / r / shape[dim - 1];  // Initial matrix row
    size_t col = r * shape[dim - 1];         // Initial matrix column
    COOMatrix_l2<T> W = tensor.reshape2Mat(row, col);

    for (int i = dim - 1; i > 0; i--) {
        // Reshape matrix (skip the first iteration)
        if (i != dim - 1) {
            row = nbar / r / shape[i];  // Matrix row
            col = r * shape[i];         // Matrix column
            W.reshape(row, col);
        }

        // Sparse interpolative decomposition
        auto idResult = dSparse_Interpolative_CPU(W, cutoff, spthres, r_max);
        
        // ...
        


    }

    // Memory release...

    return;
}

// Explicitly instantiate the specializations
template void TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, size_t, double, bool);
template void TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, size_t, double, bool);
template void TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, size_t, double, bool);
template void TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, size_t, double, bool);

