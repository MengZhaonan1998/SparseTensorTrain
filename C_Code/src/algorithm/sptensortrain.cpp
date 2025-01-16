#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/functions.h"
#include "new/structures.h"
#include "new/util.h"

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
        std::cout << "TT ITERATION " << i << std::endl;
        
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
        
        // Print the low rank approximation error
        if (verbose) {            
            auto C = W.subcol(idResult.pivot_cols, ri);
            auto Z = dcoeffZRecon(idResult.interp_coeff, idResult.pivot_cols, ri, col);            
            // There are bugs in .multiply function
            //auto recon = C.multiply(Z);
            
            std::cout << "C\n";
            //util::PrintMatWindow(C.todense(), C.rows, C.cols, {0, C.rows-1}, {0, C.cols-1});
            C.print();
            std::cout << "Z\n";
            //util::PrintMatWindow(Z.todense(), Z.rows, Z.cols, {0, Z.rows-1}, {0, Z.cols-1});
            Z.print();

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
        // TODO...

        // Form new W from the interpolative factor C
        nbar = nbar * ri / shape[i] / r;  // New total size of W
        r = ri;
        W = W.subcol(idResult.pivot_cols, ri);
    }

    // Memory release...

    return;
}

// Explicitly instantiate the specializations
template void TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, double, double, size_t, bool);
template void TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, double, double, size_t, bool);
template void TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, double, double, size_t, bool);
template void TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, double, double, size_t, bool);