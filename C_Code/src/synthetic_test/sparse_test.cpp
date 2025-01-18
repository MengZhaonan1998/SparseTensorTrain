#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/util.h"
#include "new/functions.h"
#include "new/structures.h"

void test1()
{
    // Initialize the synthetic sparse tensor
    COOTensor<double, 2> G1(10, 20, 10);    
    COOTensor<double, 3> G2(10, 10, 30, 30);
    COOTensor<double, 3> G3(10, 30, 30, 5);
    COOTensor<double, 2> G4(10, 5, 20);
    G1.generate_random(0.1, Distribution::UNIFORM, DistributionParams::uniform(0.0, 1.0), 100);
    G2.generate_random(0.05, Distribution::UNIFORM, DistributionParams::uniform(0.0, 1.0), 200);
    G3.generate_random(0.05, Distribution::UNIFORM, DistributionParams::uniform(0.0, 1.0), 300);
    G4.generate_random(0.1, Distribution::UNIFORM, DistributionParams::uniform(0.0, 1.0), 400);
    auto T = SparseTTtoTensor<double>(G1, G2, G3, G4);
    
    std::cout << "INPUT SETTINGS\n"; 
    std::cout << "Input core G1 - density: " << G1.get_density() << "\n";
    std::cout << "Input core G2 - density: " << G2.get_density() << "\n";
    std::cout << "Input core G3 - density: " << G3.get_density() << "\n"; 
    std::cout << "Input core G4 - density: " << G4.get_density() << "\n";
    std::cout << "Synthetic tensor - nnz " << T.nnz() << ", density: " << T.get_density() << "\n";


    size_t r_max = 100;
    double eps = 1e-8;
    double spthres = 0.3;
    bool verbose = true;
    
    if (1) {
        TT_ID_sparse(T, eps, spthres, r_max, verbose);
    } else {
        auto T_full = T.to_dense();
        TT_IDPRRLDU_dense(T_full, r_max, eps);
    }

    return;
}

int main(int argc, char** argv) 
{ 
    test1();
    return 0; 
} 