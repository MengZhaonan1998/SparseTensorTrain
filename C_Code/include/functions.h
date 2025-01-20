// functions.h - Matrix decomposition / Tensor train functions
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "core.h"
#include "external.h"
#include "structures.h"
#include "spmatrix.h"

// BLAS operations
void blas_dcolumn_inner_products(const double* A, int m, int n, double* results);

// SVD related functions
void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);

// QR related functions
void dQR_MGS(double* M, int Nr, int Nc, double* Q, double* R);
double verifyQR(int m, int n, double* Q, double* R, double* A, int* jpvt);
void dPivotedQR(int m, int n, double* A, double* Q, double* R, int* jpvt);
void dPivotedQR_MGS(double* M, int Nr, int Nc, double* Q, double* R, int* P, int& rank);

// Partial rank-revealing LDU decomposition
decompRes::PrrlduRes<double> 
dPartialRRLDU(double* M_, size_t Nr, size_t Nc, double cutoff, size_t maxdim, size_t mindim);

// Interpolative decomposition
void dInterpolative_PivotedQR(double* A, int m, int n, int maxdim, double* C, double* Z, int& outdim);
void dInterpolative_PrrLDU(double* M, size_t Nr, size_t Nc, size_t maxdim, double cutoff, double* C, double* Z, size_t& outdim);

// Tensor decomposition functions
std::vector<tblis::tensor<double>> TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);
std::vector<tblis::tensor<double>> TT_IDQR_dense_nocutoff(tblis::tensor<double> tensor, int r_max);
std::vector<tblis::tensor<double>> TT_IDPRRLDU_dense(tblis::tensor<double> tensor, int r_max, double eps);

// Sparse decompositions
decompRes::SparsePrrlduRes<double>
dSparse_PartialRRLDU_CPU(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, size_t const maxdim, bool const isFullReturn);
decompRes::SparseInterpRes<double>
dSparse_Interpolative_CPU(COOMatrix_l2<double> const M_, double const cutoff, double const spthres, size_t const maxdim);
COOMatrix_l2<double> dcoeffZRecon(double* coeffMatrix, size_t* pivot_col, size_t rank, size_t col);

#endif // TENSOR_DECOMPOSITION_H