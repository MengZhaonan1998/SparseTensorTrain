#include "header.h"

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

int main(int argc, char** argv) 
{
    std::cout << "Synthetic test starts!" << std::endl;
    synthetic_test_1();
    std::cout << "Synthetic test ends!" << std::endl;
    return 0; 
}