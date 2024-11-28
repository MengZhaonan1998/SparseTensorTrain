#include "new/core.h"
#include "new/utils.h"
#include "new/functions.h"

enum dense_tt_id {
    TTSVD = 1,
    TTID_PRRLDU = 2,
    TTID_RRQR = 3
};

void SyntheticDenseTest(std::initializer_list<int> tShape, std::initializer_list<int> tRank, 
                        dense_tt_id ttalgoType)
{
    std::cout << "Synthetic dense test starts.\n"; 
    
    // Generate a synthetic dense tensor {tShape} (TT rank = {tRank})
    auto synTensor = util::SyntheticTenGen<double>(tShape, tRank);

    // Tensor train decomposition
    int maxdim = std::max(tRank);
    double cutoff = 1E-10; 
    std::vector<tblis::tensor<double>> ttList;
    switch (ttalgoType) {
    case TTSVD:
        std::cout << "TT-SVD starts\n";
        ttList = TT_SVD_dense(synTensor, maxdim, cutoff);
        std::cout << "TT-SVD ends.\n";
        break;
    case TTID_PRRLDU:
        std::cout << "TT-ID-PRRLDU starts\n";
        ttList = TT_IDPRRLDU_dense(synTensor, maxdim, cutoff);
        std::cout << "TT-ID-PRRLDU ends.\n";
        break;
    case TTID_RRQR:
        std::cout << "...TDID_RRQR is to be debugged...\n";
        break;
    default:
        std::cout << "Please give a correct no. of TT algorithm types.\n";
        break;
    }   

    // Reconstruction evaluation
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(synTensor, recTensor, 2, true);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Test ends." << std::endl;
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
    //std::cout << "The toy tensor is \n" << tensor << std::endl;
    auto ttList = TT_IDPRRLDU_dense(tensor, 100, 1e-10);
    auto recTensor = util::TT_Contraction_dense(ttList);
    double error = util::NormError(tensor, recTensor, 2, false);
    std::cout << "TT recon error: " << error << "\n";
    std::cout << "Synthetic test 1 ends." << std::endl;
}

int main(int argc, char** argv) 
{
    SyntheticDenseTest({20, 30, 30, 10}, {13, 20, 7}, TTID_PRRLDU);
    return 0; 
}


