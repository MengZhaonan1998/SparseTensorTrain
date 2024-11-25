// decomposition.h - Matrix decomposition functions
#ifndef TENSOR_DECOMPOSITION_H
#define TENSOR_DECOMPOSITION_H

#include "core.h"
#include "external.h"
#include "structures.h"

// SVD related functions
void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);

// QR related functions
void dQR_MGS(double* M, int Nr, int Nc, double* Q, double* R);
double verifyQR(int m, int n, double* Q, double* R, double* A, int* jpvt);
void dPivotedQR(int m, int n, double* A, double* Q, double* R, int* jpvt);
void dPivotedQR_MGS(double* M, int Nr, int Nc, double* Q, double* R, int* P, int& rank);
void dInterpolative_PivotedQR(double* A, int m, int n, int maxdim, double* C, double* Z, int& outdim);

// BLAS operations
void blas_dcolumn_inner_products(const double* A, int m, int n, double* results);

// Tensor decomposition functions
std::vector<tblis::tensor<double>> TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);
std::vector<tblis::tensor<double>> TT_IDQR_dense_nocutoff(tblis::tensor<double> tensor, int r_max);

// Partial decomposition
decompRes::PrrlduRes<double> dPartialRRLDU(double* M_, size_t Nr, size_t Nc,
                                          double cutoff, size_t maxdim, size_t mindim);

#endif // TENSOR_DECOMPOSITION_H