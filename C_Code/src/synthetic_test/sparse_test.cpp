#include "new/spmatrix.h"
#include "new/sptensor.h"
#include "new/util.h"
#include "new/functions.h"
#include "new/structures.h"

void test1()
{
    // Initialize the synthetic sparse tensor
    util::Timer timer("test1");
    COOTensor<double, 2> G1(100, 20, 10);    
    COOTensor<double, 3> G2(100, 10, 25, 30);
    COOTensor<double, 3> G3(100, 30, 50, 9);
    COOTensor<double, 2> G4(100, 9, 11);
    G1.generate_random(0.1, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 100);
    G2.generate_random(0.03, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 200);
    G3.generate_random(0.03, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 300);
    G4.generate_random(0.05, Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), 400);
    auto T = SparseTTtoTensor<double>(G1, G2, G3, G4);
    
    std::cout << "INPUT SETTINGS\n"; 
    std::cout << "Input core G1 - density: " << G1.get_density() << "\n";
    std::cout << "Input core G2 - density: " << G2.get_density() << "\n";
    std::cout << "Input core G3 - density: " << G3.get_density() << "\n"; 
    std::cout << "Input core G4 - density: " << G4.get_density() << "\n";
    std::cout << "Synthetic tensor - nnz " << T.nnz() << ", density: " << T.get_density() << "\n";
    
    size_t r_max = 30;
    double eps = 1e-8;
    double spthres = 0.3;
    bool verbose = false;
    
    if (1) {
        std::cout << "TT_ID_sparse\n";
        auto ttList = TT_ID_sparse(T, eps, spthres, r_max, verbose);

        std::cout << "fac 1\n";
        ttList.StartG.print();

        std::cout << "fac 2\n";
        ttList.InterG[0].print();

        std::cout << "fac 3\n";
        ttList.InterG[1].print();

        std::cout << "fac 4\n";
        ttList.EndG.print();

        //auto reconT = SparseTTtoTensor<double>(ttList.StartG, ttList.InterG[0], ttList.InterG[1], ttList.EndG);
        //reconT.print();

        //double err = T.rel_diff(reconT);
        //std::cout << "error: " << err << std::endl;
        

    } else {
        std::cout << "TT_IDPRRLDU_dense\n";
        auto T_full = T.to_dense();
        auto ttList = TT_IDPRRLDU_dense(T_full, r_max, eps);
    }

    return;
}

int main(int argc, char** argv) 
{ 
    test1();
    util::Timer::summarize();
    return 0; 
} 