#include "new/core.h"
#include "new/utils.h"
#include "new/functions.h"

void synthetic_test_1()
{   std::cout << "Synthetic test 1 (dense TT, dense SVD) starts." << "\n";
    auto synTensor = util::SyntheticTenGen<double>({4,5,7,5},{2,3,4});
    auto ttList = TT_SVD_dense(synTensor, 5, 1E-10);
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(synTensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
    return;
}

void synthetic_test_2()
{   std::cout << "Synthetic test 2 (dense TT, dense IDQR) starts." << "\n";
    auto synTensor = util::SyntheticTenGen<double>({15,5,10,7,20},{3,7,6,10});
    auto ttList = TT_IDQR_dense_nocutoff(synTensor, 10);
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(synTensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
    return;
}

void toy_test()
{
    // Initialize a very simple tensor
    int d1 = 5, d2 = 10, d3 = 6, d4 = 7, d5 = 11;
    tblis::tensor<double> tensor({d1, d2, d3, d4, d5});
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j)
            for (int m = 0; m < d3; ++m)
                for (int n = 0; n < d4; ++n)
                    for (int l = 0; l < d5; ++l)
                        tensor(i, j, m, n) = i * j - m + n * l;
    std::cout << "The toy tensor is \n" << tensor << std::endl;
    auto ttList = TT_IDQR_dense_nocutoff(tensor, 100);
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(tensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
}

int main(int argc, char** argv) 
{
    std::cout << "Synthetic test starts!" << std::endl;
    //toy_test();
    synthetic_test_2();
    std::cout << "Synthetic test ends!" << std::endl;
    return 0; 
}


