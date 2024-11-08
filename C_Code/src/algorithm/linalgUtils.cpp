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

void dPivotedQR(int m, int n, double* A, double* Q, double* R, int* jpvt)
{
    // Lapacke QR: dgeqp3
    int info;    
    double* tau = new double[std::min(m,n)];  // Scalar factors for elementary reflectors
    info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
    assert(info == 0);  // Check if query was successful

    // Construct R matrix from the upper triangular part of A
    for (int i = 0; i < m; ++i) 
        for (int j = i; j < n; ++j) 
            R[i * n + j] = A[i * n + j];

    // Construct Q matrix using dorgqr
    for (int i = 0; i < m; ++i) 
        for (int j = 0; j < m; ++j) 
            Q[i * m + j] = A[i * n + j];  // Copy the first m columns of A into Q
    // Use Lapacke's dorgqr to generate Q from A and tau
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, m, m, Q, m, tau);
    assert(info == 0);  // Check if Q generation was successful
    delete[] tau;
}

// Helper function to compute and verify QR - A
double verifyQR(int m, int n, double* Q, double* R, double* A, int* jpvt) {
    // Reconstruct A from Q and R
    double error = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            for (int k = 0; k < m; ++k) {
                reconstructed += Q[i * m + k] * R[k * n + j];
            }
            error += std::pow(reconstructed - A[i * n + jpvt[j] - 1], 2);
        }
    }
    error = std::sqrt(error);
    //std::cout << "Reconstruction error ||QR - A||_F = " << error << std::endl;
    return error;
}

void qr_decomp_mgs(double* M, int Nr, int Nc, double* Q, double* R) {
    for (int j = 0; j < Nc; j++) {
        // Compute the j-th column of Q
        for (int i = 0; i < Nr; i++) {
            Q[i * Nc + j] = M[i * Nc + j];
        }

        for (int i = 0; i < j; i++) {
            double dot_product = 0.0;
            for (int k = 0; k < Nr; k++) {
                dot_product += Q[k * Nc + i] * Q[k * Nc + j];
            }
            R[i * Nc + j] = dot_product;
            for (int k = 0; k < Nr; k++) {
                Q[k * Nc + j] -= R[i * Nc + j] * Q[k * Nc + i];
            }
        }

        double norm = 0.0;
        for (int k = 0; k < Nr; k++) {
            norm += Q[k * Nc + j] * Q[k * Nc + j];
        }
        norm = std::sqrt(norm);
        R[j * Nc + j] = norm;
        for (int k = 0; k < Nr; k++) {
            Q[k * Nc + j] /= norm;
        }
    }
}

void dInterpolative_qr(double* M, int m, int n, int maxdim, double* C, double* Z)
{
    int max_rank = m <= n ? m : n;
    int k = maxdim > max_rank ? max_rank : maxdim;
    double* Q = new double[m * m];
    double* R = new double[m * n]{0.0};
    int* jpvt = new int[n];       // Pivot indices

    dPivotedQR(m, n, M, Q, R, jpvt); 

    // Slice R -> R_k
    double* R_k = new double[k * k];
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            R_k[i * k + j] = R[i * n + j];
        }
    //.....TODO.....
    

    delete[] Q;
    delete[] R;
    delete[] R_k;
    return;
}