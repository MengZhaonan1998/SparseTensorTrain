// structures.h - Data structures and result types
#ifndef STRUCTURES_H
#define STRUCTURES_H

#include "core.h"
#include "spmatrix.h"

namespace decompRes {
    template<class T>
    struct PrrlduRes {
        T* L = nullptr;
        T* d = nullptr;
        T* U = nullptr;
        size_t rank;
        size_t* row_perm_inv = nullptr;
        size_t* col_perm_inv = nullptr;
        T inf_error;
        
        // Memory release
        void freeLduRes() {
            if (L != nullptr) delete[] L;
            if (d != nullptr) delete[] d;
            if (U != nullptr) delete[] U;
            if (row_perm_inv != nullptr) delete[] row_perm_inv;
            if (col_perm_inv != nullptr) delete[] col_perm_inv;
        };
    };

    template<class T>
    struct SparsePrrlduRes {
        T* d = nullptr;
        T* dense_L = nullptr;
        T* dense_U = nullptr;
        COOMatrix_l2<T>* sparse_L = nullptr;
        COOMatrix_l2<T>* sparse_U = nullptr;

        bool isSparseRes;
        bool isAllReturn;
        size_t rank;
        size_t* row_perm_inv = nullptr;
        size_t* col_perm_inv = nullptr;
        T inf_error;
        
        // Memory release
        void freeSpLduRes() {
            if (d != nullptr) delete[] d;
            if (dense_L != nullptr) delete[] dense_L;
            if (dense_U != nullptr) delete[] dense_U;
            if (sparse_L != nullptr) delete sparse_L;
            if (sparse_U != nullptr) delete sparse_U;
            if (row_perm_inv != nullptr) delete[] row_perm_inv;
            if (col_perm_inv != nullptr) delete[] col_perm_inv;
        };
    };
}

enum class Distribution {
    UNIFORM,      // Uniform distribution between min and max
    NORMAL,       // Normal distribution with mean and standard deviation
    STANDARD_NORMAL,  // Normal distribution with mean=0, std=1
    GAMMA         // Gamma distribution with shape (k) and scale (theta)
};

struct DistributionParams {
    // Parameters for various distributions
    double min_value = -1.0;     // For uniform
    double max_value = 1.0;      // For uniform
    double mean = 0.0;           // For normal
    double std_dev = 1.0;        // For normal
    double gamma_shape = 1.0;    // k (shape) parameter for gamma
    double gamma_scale = 1.0;    // θ (scale) parameter for gamma
    
    // Constructor for uniform distribution
    static DistributionParams uniform(double min = -1.0, double max = 1.0) {
        DistributionParams params;
        params.min_value = min;
        params.max_value = max;
        return params;
    }
    
    // Constructor for normal distribution
    static DistributionParams normal(double mean = 0.0, double std_dev = 1.0) {
        DistributionParams params;
        params.mean = mean;
        params.std_dev = std_dev;
        return params;
    }
    
    // Constructor for gamma distribution
    static DistributionParams gamma(double shape = 1.0, double scale = 1.0) {
        DistributionParams params;
        params.gamma_shape = shape;
        params.gamma_scale = scale;
        return params;
    }
};

#endif // TENSOR_STRUCTURES_H