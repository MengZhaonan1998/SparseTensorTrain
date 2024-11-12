#include "header.h"

void blas_dcolumn_inner_products(const double* A, int m, int n, double* results) {
    for (int i = 0; i < n; ++i) {
        // Extract column i and compute its inner product with itself
        double inner_product = cblas_ddot(m, &A[i], n, &A[i], n);
        results[i] = inner_product;
    }
    return;
}

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

void dPivotedQR_MGS(double* M, int Nr, int Nc, double* Q, double* R, double* P)
{   
    // v_j = ||X[:,j]||^2, j=1,...,n
    double* v = new double[Nc]{0.0};
    blas_dcolumn_inner_products(M, Nr, Nc, v);

    // Determine an index p1 such that v_p1 is maximal
    double* max_ptr_v = std::max_element(v, v + Nc);
    int pk = std::distance(v, max_ptr_v);  

    // Initialization of arrays
    std::iota(P, P + Nc, 0);        // Fill the permutation array with 0, 1, 2, ..., Nc.
    std::fill(Q, Q + Nc * Nr, 0.0); // Fill Q with zeros
    std::fill(R, R + Nc * Nc, 0.0); // Fill R with zeros
    //util::Print1DArray(v, Nc);

    // Modified Gram-Schmidt Process (To be modified? MGS)
    int rank = 0;
    for (int k = 0; k < Nc; ++k) {
        // Swap arrays: X, v, P, R 
        cblas_dswap(Nr, M + pk, Nc, M + k, Nc); // Swap the pk-th and j-th column of M (To be optimized?)
        cblas_dswap(1, v + pk, 1, v + k, 1);    
        cblas_dswap(1, P + pk, 1, P + k, 1);
        cblas_dswap(k, R + pk, Nc, R + k, Nc);    

        // I can use blas but I write my own code here for future optimization
        for (int i = 0; i < Nr; ++i) {
            double temp = 0.0;
            for (int j = 0; j < k; ++j) 
                temp += Q[i * Nc + j] * R[j * Nc + k];
            Q[i * Nc + k] = M[i * Nc + k] - temp;
        }

        double inner_prod = 0.0;
        for (int i = 0; i < Nr; ++i) 
            inner_prod += Q[i * Nc + k] * Q[i * Nc + k];
        R[k * Nc + k] = std::sqrt(inner_prod);

        for (int i = 0; i < Nr; ++i) 
            Q[i * Nc + k] = Q[i * Nc + k] / R[k * Nc + k];

        for (int i = k + 1; i < Nc; ++i) {
            double temp = 0.0;
            for (int j = 0; j < Nr; ++j) 
                temp += Q[j * Nc + k] * M[j * Nc + i];
            R[k * Nc + i] = temp;
        }
        
        // Rank increment
        rank += 1;    

        // Update v_j
        for (int j = k + 1; j < Nc; ++j) 
            v[j] = v[j] - R[k * Nc + j] * R[k * Nc + j];

        // Determine an index p1 such that v_p1 is maximal
        max_ptr_v = std::max_element(v + k + 1, v + Nc);
        pk = std::distance(v, max_ptr_v);  

        // Rank revealing step
        if (v[pk] < 1E-10) 
            break;
    }

    delete[] v;
    return;
}

    //std::cout << "Iter:" << k << std::endl;
    //std::cout << "Q:" << std::endl;
    //util::PrintMatWindow(Q, Nr, Nc, {0,Nr-1}, {0,Nc-1});
    //std::cout << "R:" << std::endl;
    //util::PrintMatWindow(R, Nc, Nc, {0,Nc-1}, {0,Nc-1});

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