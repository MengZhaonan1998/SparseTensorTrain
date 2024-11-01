#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <lapacke.h>
#include <tblis/tblis.h>

void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);
void TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);

#endif