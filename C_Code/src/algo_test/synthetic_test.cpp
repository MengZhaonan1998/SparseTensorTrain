
#include "header.h"
/*
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
*/

void swap_columns(double* A, int m, int n, int col1, int col2) {
    // Check if column indices are within valid bounds
    if (col1 >= n || col2 >= n || col1 < 0 || col2 < 0) {
        throw std::out_of_range("Error: Column indices out of bounds.");
    }

    // Swap columns col1 and col2 using cblas_dswap
    // Since A is in row-major, we treat columns as vectors and swap them one-by-one
    //for (int i = 0; i < m; ++i) {
        // Swap elements in the same row for the two columns
    //    cblas_dswap(m, &A[i * n + col1], n, &A[i * n + col2], n);
    //}
    cblas_dswap(m, A + col1, n, A + col2, n);
}

int main() {
    
    int m = 4; // Number of rows
    int n = 3; // Number of columns
    double A[12] = {
        1.0, 2.0, 3.0,  // Row 0
        4.0, 5.0, 6.0,  // Row 1
        7.0, 8.0, 9.0,  // Row 2
        10.0, 11.0, 12.0 // Row 3
    };

    double* Q = new double[m * n];
    double* R = new double[n * n];
    double* P = new double[n];
    dPivotedQR_MGS(A, m, n, Q, R, P);


    return 0;
}