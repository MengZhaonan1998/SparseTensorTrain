#include "header.h"

void svd(double* A, int m, int n, double* S, double* U, double* VT) {
    // Work array size determination
    int lda = m, ldu = m, ldvt = n;
    int lwork = -1;
    double wkopt;
    int info;

    // Query optimal workspace size
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, &wkopt, lwork);
    lwork = (int)wkopt;
    double* work = new double[lwork];

    // Compute SVD
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork);

    if (info > 0) {
        // Handle error in SVD computation
        std::cerr << "SVD failed to converge." << std::endl;
    }

    // Clean up
    delete[] work;
}

int main(int argc, char** argv) 
{
    std::cout << "Test of synthetic tensors starts!" << std::endl;
    
    return 0; 
}