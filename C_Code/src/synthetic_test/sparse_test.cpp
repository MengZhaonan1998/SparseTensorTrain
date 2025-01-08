#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/util.h"
#include "new/functions.h"
#include "new/structures.h"

int main(int argc, char** argv) 
{
    /*
    COOTensor<double, 3> tensor3d(100, 5, 5, 5); // capacity, dim1, dim2, dim3
    
    size_t r_max = 10;
    double eps = 1e-8;
    bool verbose = true;
    TT_ID_sparse(tensor3d, r_max, eps, verbose);  // double, 3
    */

   
   /*
   M = np.array([[0.0, 2.0, 0.0, -3.0, 1.2],
              [0.0, 0.0, -0.5, 0.0, 4.0],
              [-1.5, 0.0, 0.0, 5.0, 0.0],
              [0.0, -9.9, 0.2, 0.0, 0.0]])
    */
    
    COOMatrix_l2<double> M_(4, 5, 20);
    M_.add_element(0, 1, 2.0);
    M_.add_element(0, 3, -3.0);
    M_.add_element(0, 4, 1.2);
    M_.add_element(1, 2, -0.5);
    M_.add_element(1, 4, 4.0);
    M_.add_element(2, 0, -1.5);
    M_.add_element(2, 3, 5.0);
    M_.add_element(3, 1, -9.9);
    M_.add_element(3, 2, 0.2);
    
    //util::PrintMatWindow(M_.todense(), 4, 5, {0,3},{0,4});
    bool isFullReturn = true;
    double cutoff = 1e-8;
    size_t maxdim = 5;
    size_t mindim = 1;
    auto result = dSparse_PartialRRLDU_CPU(M_, cutoff, maxdim, mindim, isFullReturn);



    return 0; 
} 