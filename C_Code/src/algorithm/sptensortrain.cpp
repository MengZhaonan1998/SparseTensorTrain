#include "new/spmatrix.h"
#include "new/sptensor.h"



// Define the template function TT_ID_sparse
template<typename T, size_t Order>
void TT_ID_sparse(const COOTensor<T, Order>& tensor, size_t r_max, T eps, bool verbose) {
    
    auto shape = tensor.get_dimensions();  // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();                // Get the number of dimension d    
    size_t nbar = 1;                       // Get the total size: n1 * n2 * ... * nd 
    for (int i = 0; i < dim; ++i)
        nbar *= shape[i];

    size_t r = 1;  // Rank
    for (int i = dim - 1; i > 0; i--) {
        int row = nbar / r / shape[i];     // Reshape row
        int col = r * shape[i];            // Reshape column
        
    }



    return;
    
}

// Explicitly instantiate the specializations
template void TT_ID_sparse<double, 3>(const COOTensor<double, 3>&, size_t, double, bool);
template void TT_ID_sparse<double, 4>(const COOTensor<double, 4>&, size_t, double, bool);
template void TT_ID_sparse<double, 5>(const COOTensor<double, 5>&, size_t, double, bool);
template void TT_ID_sparse<double, 6>(const COOTensor<double, 6>&, size_t, double, bool);

