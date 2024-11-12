#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <random>
#include <cmath>
#include <algorithm>
#include <cblas.h>
#include <lapacke.h>
#include <tblis/tblis.h>

void blas_dcolumn_inner_products(const double* A, int m, int n, double* results);
void fSVD(float* A, int m, int n, float* S, float* U, float* VT);
void dSVD(double* A, int m, int n, double* S, double* U, double* VT);
void dPivotedQR(int m, int n, double* A, double* Q, double* R, int* jpvt);
double verifyQR(int m, int n, double* Q, double* R, double* A, int* jpvt);
void dInterpolative_QR(double* M, int m, int n, int maxdim, double* C, double* Z);

void qr_decomp_mgs(double* M, int Nr, int Nc, double* Q, double* R);
void dPivotedQR_MGS(double* M, int Nr, int Nc, double* Q, double* R, double* P);


std::vector<tblis::tensor<double>> TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);

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
size_t GetSize(tblis::tensor<T> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    size_t len = 1;
    for (int i=0; i<dim; ++i) len *= shape[i];
    return len;
}

template<class T>
double Norm(tblis::tensor<T> tensor, int mode)
{
    if (mode != 1 && mode != 2) 
        throw std::invalid_argument("Incorrect mode! Mode should be either 1 (max norm) or 2 (frob2 norm).");
    double norm = 0.0;
    size_t N = GetSize(tensor);
    if (mode == 1) {
        // Mode 1: max norm
        auto data = tensor.data();
        for (size_t i = 0; i < N; ++i) 
            norm = std::max(std::abs(data[i]), norm);
    } else if (mode == 2) {
        // Mode 2: Frobenious 2 norm 
        auto data = tensor.data();
        for (size_t i = 0; i < N; ++i) 
            norm += data[i] * data[i];
        norm = std::sqrt(norm);
    }    
    return norm;
}

template<class T>
void PrintMatWindow(T* matrix, size_t row, size_t col,  
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

template<class T>
void Print1DArray(T* array, size_t N) 
{
    std::cout << "[" << array[0];
    for (size_t i = 1; i < N; ++i) {
        std::cout << ", " << array[i];
    }
    std::cout << "]" << std::endl;
    return;
}

inline std::string generateLetters(char offset, int n) {
    std::string result;
    for (int i = 0; i < n; ++i) {
        result += offset + i;  // Append the next letter in the sequence
    }
    return result;
}

template<class T>
tblis::tensor<T> TT_Contraction_dense(std::vector<tblis::tensor<T>> ttList)
{
    int recon_dim = ttList.size();
    tblis::tensor<T> tensor = ttList[0];
    for (int i = 1; i < recon_dim; ++i) {
        auto factor = ttList[i];
        auto lshape = tensor.lengths();
        auto rshape = factor.lengths();
        int ldim = lshape.size();
        int rdim = rshape.size();        
        MArray::len_vector ishape(ldim + rdim - 2);

        for (int i = 0; i < ldim + rdim - 2; ++i) {
            if (i < ldim - 1) {
                ishape[i] = lshape[i];
            } else {
                ishape[i] = rshape[i - ldim + 2];
            }
        }
        std::string aidx = generateLetters('a', ldim);
        std::string bidx = generateLetters(aidx.back(), rdim);
        std::string cidx = generateLetters('a', ldim + rdim - 1);
        cidx.erase(ldim-1, 1);

        tblis::tensor<T> inter_tensor(ishape);
        tblis::mult<T>(1.0, tensor, aidx.c_str(), factor, bidx.c_str(), 0.0, inter_tensor, cidx.c_str());  
        tensor.resize(ishape); 
        tensor = inter_tensor;
    }
    return tensor;
}

template<class T>
tblis::tensor<T> SyntheticTenGen(std::initializer_list<int> tShape, std::initializer_list<int> tRank)
{
    if (tRank.size() != tShape.size() - 1) {
        throw std::invalid_argument("Invalid input tShape or tRank!");
    }
    
    std::vector<int> vec_tShape = tShape;
    std::vector<int> vec_tRank = tRank;
    vec_tRank.push_back(1);
    vec_tRank.insert(vec_tRank.begin(), 1);
    int dim = vec_tShape.size();
    std::vector<tblis::tensor<T>> ttList;
    
    // Tensor train generation
    for (int i = 0; i < dim; ++i) {
        int n = vec_tRank[i] * vec_tShape[i] * vec_tRank[i+1];
        tblis::tensor<T> factor({vec_tRank[i], vec_tShape[i], vec_tRank[i+1]});
        if (i == 0) { 
            factor.resize({vec_tShape[i], vec_tRank[i+1]}); }
        else if (i == dim - 1) { 
            factor.resize({vec_tRank[i], vec_tShape[i]}); }
        
        // Generate the random data for the tensor factor
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10,10);
        T* randSeq = new T[n];
        for (int i = 0; i < n; ++i) randSeq[i] = dis(gen);
        std::copy(randSeq, randSeq + n, factor.data());       
        ttList.push_back(factor);
        delete[] randSeq;    
    }
    // TT contraction -> Synthetic output tensor
    auto synTensor = util::TT_Contraction_dense(ttList);
    return synTensor;
}

template<class T>
double NormError(tblis::tensor<T> tensorA, tblis::tensor<T> tensorB, int mode, bool relative)
{
    size_t N = GetSize(tensorA);
    if (N != GetSize(tensorB))
        throw std::invalid_argument("Size of tensorA != Size of tensorB!");
    if (mode != 1 && mode != 2) 
        throw std::invalid_argument("Incorrect mode! Mode should be either 1 (max-norm error) or 2 (frob-norm error).");
    
    double error = 0.0;
    if (mode == 1) {
        // Mode 1: max-norm error
        auto tensorA_data = tensorA.data();
        auto tensorB_data = tensorB.data();
        for (size_t i = 0; i < N; ++i) 
            error = std::max(error, std::abs(tensorA_data[i] - tensorB_data[i]));      
    } else if (mode == 2) {
        // Mode 2: frobenious norm-2 error
        auto tensorA_data = tensorA.data();
        auto tensorB_data = tensorB.data();
        for (size_t i = 0; i < N; ++i) {
            auto diff = tensorA_data[i] - tensorB_data[i];
            error += diff * diff;
        }
        error = std::sqrt(error);
    }
    if (relative) 
        error /= Norm(tensorA, mode);
    
    return error;
}

}

#endif