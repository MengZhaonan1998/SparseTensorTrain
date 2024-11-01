#include "header.h"

void toy_test()
{   
    tblis::tensor<double> A({2, 4, 3});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                A(i,j,k) = i + j + k;
    
    auto aa = A.lengths();
    std::cout << aa[0] << " " << aa[1] << " " << aa[2] << std::endl;

    TT_SVD_dense(A, 5, 1E-5);

}

int main(int argc, char** argv) 
{
    std::cout << "Test of synthetic tensors starts!" << std::endl;
    toy_test();
    return 0; 
}