#include "header.h"

void fSVD(float* A, int m, int n, float* S, float* U, float* VT) {
    int lda = n;      // Leading dimension of A
    int ldu = m;      // Leading dimension of U
    int ldvt = n;     // Leading dimension of VT
    int info;
    // Query for optimal workspace size
    float work_size;
    info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, &work_size, -1);
    if (info != 0) throw std::runtime_error("SVD workspace query failed.");
    // Allocate optimal workspace
    int lwork = static_cast<int>(work_size);
    std::vector<float> work(lwork);
    // Call the LAPACKE SVD function with the allocated workspace
    info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work.data(), lwork);
    if (info > 0) throw std::runtime_error("SVD computation did not converge.");
}

void dSVD(double* A, int m, int n, double* S, double* U, double* VT) {
    int lda = n;      // Leading dimension of A
    int ldu = m;      // Leading dimension of U
    int ldvt = n;     // Leading dimension of VT
    int info;
    // Query for optimal workspace size
    double work_size;
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, &work_size, -1);
    if (info != 0) throw std::runtime_error("SVD workspace query failed.");
    // Allocate optimal workspace
    int lwork = static_cast<int>(work_size);
    std::vector<double> work(lwork);
    // Call the LAPACKE SVD function with the allocated workspace
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work.data(), lwork);
    if (info > 0) throw std::runtime_error("SVD computation did not converge.");
}

void ttsvd()
{

    return;    
}
