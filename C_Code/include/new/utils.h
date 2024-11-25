// tensor_utils.h - Tensor utility functions
#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "core.h"
#include "external.h"

namespace util {
    template<class T>
    T FrobNorm(tblis::tensor<T> tensor);

    template<class T>
    size_t GetSize(tblis::tensor<T> tensor);

    template<class T>
    double Norm(tblis::tensor<T> tensor, int mode);

    template<class T>
    void PrintMatWindow(T* matrix, size_t row, size_t col,  
                       std::tuple<int,int> rmask, std::tuple<int,int> cmask);

    template<class T>
    void Print1DArray(T* array, size_t N);

    template<class T>
    void generateRandomArray(T* array, int size, T minValue, T maxValue);

    std::string generateLetters(char offset, int n);

    template<class T>
    tblis::tensor<T> TT_Contraction_dense(std::vector<tblis::tensor<T>> ttList);

    template<class T>
    tblis::tensor<T> SyntheticTenGen(std::initializer_list<int> tShape, 
                                    std::initializer_list<int> tRank);

    template<class T>
    double NormError(tblis::tensor<T> tensorA, tblis::tensor<T> tensorB, 
                    int mode, bool relative);
}

// Template implementations
#include "utils.impl.h"

#endif // TENSOR_UTILS_H