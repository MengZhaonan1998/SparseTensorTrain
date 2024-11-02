#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <lapacke.h>
#include <tblis/tblis.h>

void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);
void TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);

namespace util
{

template<class T>
T FrobNorm(tblis::tensor<T> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    int len = 1;
    double fnorm = 0.0;
    for (int i=0; i<dim; ++i) len *= shape[i];
    auto data = tensor.data();
    for (int i=0; i<len; ++i) fnorm += data[i] * data[i];
    fnorm = std::sqrt(fnorm);
    return fnorm;
}

template<class T>
int GetSize(tblis::tensor<T> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    int len = 1;
    for (int i=0; i<dim; ++i) len *= shape[i];
    return len;
}

template<class T>
void printMatWindow(T* matrix, size_t row, size_t col,  
                    std::tuple<int,int> rmask, std::tuple<int,int> cmask)
{
    if (std::get<0>(rmask) < 0 || std::get<1>(rmask) >= row ||
        std::get<0>(cmask) < 0 || std::get<1>(cmask) >= col) {
            throw std::invalid_argument("Invalid input row or column mask.");
    }
    for (int i = std::get<0>(rmask); i <= std::get<1>(rmask); ++i) {
        for (int j = std::get<0>(cmask); j <= std::get<1>(cmask); ++j) {
            std::cout << matrix[i * col + j] << " ";        
        }
        std::cout << "\n";
    }
    return;
}

}

#endif