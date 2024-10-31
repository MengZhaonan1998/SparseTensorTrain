#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <lapacke.h>

void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);

#endif