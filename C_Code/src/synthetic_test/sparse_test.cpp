#include "spmatrix.h"
#include "sptensor.h"
#include "util.h"
#include "functions.h"
#include "structures.h"

int main(int argc, char** argv) 
{ 
    // Main process timer
    {util::Timer timer("Main process");
    
    // Config settings
    int d = 4;
    size_t shape[d] = {50, 50, 50, 50};
    size_t rank[d-1] = {30, 100, 30};
    size_t seed[d] = {100, 200, 300, 400};
    double density[d] = {1e-2, 1e-3, 1e-3, 1e-2};
    
    // Synthetic factor -> Synthetic tensor
    COOTensor<double, 2> G1(shape[0]* rank[0] * density[0], shape[0], rank[0]);    
    COOTensor<double, 3> G2(rank[0] * shape[1] * rank[1] * density[1], rank[0], shape[1], rank[1]);
    COOTensor<double, 3> G3(rank[1] * shape[2] * rank[2] * density[2], rank[1], shape[2], rank[2]);
    COOTensor<double, 2> G4(rank[2] * shape[3] * density[3], rank[2], shape[3]);
    G1.generate_random(density[0], Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), seed[0]);
    G2.generate_random(density[1], Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), seed[1]);
    G3.generate_random(density[2], Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), seed[2]);
    G4.generate_random(density[3], Distribution::UNIFORM, DistributionParams::uniform(0.0, 10.0), seed[3]);
    //COOTensor<double, 4> T(5000, shape[1], shape[2], shape[3]);    
    auto T = SparseTTtoTensor<double>(G1, G2, G3, G4);

    // Input information display
    std::cout << "INPUT SETTINGS:\n"; 
    std::cout << "Input core G1 --" << G1 << "\n";
    std::cout << "Input core G2 --" << G2 << "\n";
    std::cout << "Input core G3 --" << G3 << "\n"; 
    std::cout << "Input core G4 --" << G4 << "\n";
    std::cout << "Synthetic tensor T --" << T << "\n";

    // Tensor-train algorithm settings
    size_t r_max = 256;
    double eps = 1e-8;
    double spthres = 0.3;
    bool verbose = false;

    // Sparse TTID algorithm
    std::cout << "SPARSE TT-ID:\n";
    auto ttList = TT_ID_sparse(T, eps, spthres, r_max, verbose);
    
    // Output information display
    std::cout << "OUTPUT INFO:\n"; 
    std::cout << "Output core F1 --" << ttList.StartG << "\n";
    std::cout << "Output core F2 --" << ttList.InterG[0] << "\n";
    std::cout << "Output core F3 --" << ttList.InterG[1] << "\n"; 
    std::cout << "Output core F4 --" << ttList.EndG << "\n";
    auto reconT = SparseTTtoTensor<double>(ttList.StartG, ttList.InterG[0], ttList.InterG[1], ttList.EndG);
    std::cout << "Reconstructed tensor -- " << reconT << "\n";
    double err = T.rel_diff(reconT);
    std::cout << "Relative reconstruction error = " << err << std::endl;

    // Timer summary
    }util::Timer::summarize();
    return 0; 
} 