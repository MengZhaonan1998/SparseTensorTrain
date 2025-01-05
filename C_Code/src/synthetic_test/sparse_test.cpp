#include "new/sptensor.h"

int main(int argc, char** argv) 
{
    COOTensor<double, 3> tensor3d(100, 5, 5, 5); // capacity, dim1, dim2, dim3
    
    size_t r_max = 10;
    double eps = 1e-8;
    bool verbose = true;
    TT_ID_sparse(tensor3d, r_max, eps, verbose);  // double, 3
    return 0; 
} 