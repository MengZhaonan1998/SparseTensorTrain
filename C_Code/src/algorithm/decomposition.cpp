#include "header.h"
#include "struct.h"

void dQR_MGS(double* M, int Nr, int Nc, double* Q, double* R) {
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
    return;
}

void dPivotedQR_MGS(double* A, int Nr, int Nc, double* Q, double* R, int* P, int& rank)
{   
    // Copy the input matrix
    double* M = new double[Nr * Nc];
    std::copy(A, A + Nr * Nc, M);

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

    // Modified Gram-Schmidt Process (To be modified? MGS)
    rank = 0;
    for (int k = 0; k < Nc; ++k) {
        // Swap arrays: X, v, P, R 
        cblas_dswap(Nr, M + pk, Nc, M + k, Nc); // Swap the pk-th and j-th column of M (To be optimized?)
        cblas_dswap(1, v + pk, 1, v + k, 1);    // Swap v[k] <-> v[pk]
        cblas_dswap(k, R + pk, Nc, R + k, Nc);  // Swap R[0:k,pk] <-> R[0:k,k]  
        int temp = P[k];    // Swap P[k] <-> P[pk]
        P[k] = P[pk];
        P[pk] = temp;

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
        // PROBLEM! We need to find how to determine the rank cutoff tolerance!
        // Sometimes 1
        if (v[pk] < 1) 
            break;
    }

    delete[] v;
    delete[] M;
    return;
}

// Interpolative decomposition by pivoted QR
void dInterpolative_PivotedQR(double* M, int m, int n, int maxdim, 
                              double* C, double* Z, int& outdim)
{
    // Get CZ rank k
    int k = maxdim;
    
    // Pivoted (rank-revealing) QR decomposition
    double* Q = new double[m * n]{0.0};
    double* R = new double[n * n]{0.0};
    int* P = new int[n];
    int rank;
    dPivotedQR_MGS(M, m, n, Q, R, P, rank);
    k = k < rank ? k : rank;
    outdim = k;

    // R_k = R[0:k,0:k] (To be optimized)
    double* R_k = new double[k * k];
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) 
            R_k[i * k + j] = R[i * n + j];
    
    // C = M[:, cols]     TOBECONTINUED... Rank stuff...
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            C[i * k + j] = M[i * n + P[j]];

    // Solve linear systems for Z: (R_k^T * R_k) Z = C^T * M
    double* b = new double[k];
    for (int i = 0; i < n; ++i) {
        // Construct right hand side b = C^T * M[:,i]
        std::fill(b, b + k, 0.0);
        for (int j = 0; j < k; ++j) 
            for (int l = 0; l < m; ++l) 
                b[j] += C[l * k + j] * M[l * n + i];
        // Solve two triangular systems R_k/R_k^T
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, k, R_k, k, b, 1);
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, k, R_k, k, b, 1);  
        // Copy solution to Z
        for (int j = 0; j < k; ++j) 
            Z[j * n + i] = b[j];
    }

    delete[] Q;
    delete[] R;
    delete[] P;
    delete[] R_k;
    delete[] b;
    return;        
}

decompRes::PrrlduRes<double> 
dPartialRRLDU(double* M, size_t Nr, size_t Nc,
              double cutoff, size_t maxdim, size_t mindim) 
{
    decompRes::PrrlduRes<double> resultSet;

    return resultSet;
}