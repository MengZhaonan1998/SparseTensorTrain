#include <gtest/gtest.h>
#include "new/core.h"
#include "new/utils.h"
#include "new/functions.h"

TEST(QRID_TEST, low_rank_synthetic1)
{
    int m = 10;   // Rows 
    int n = 8;    // Cols
    double* M = new double[m * n];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            M[i * n + j] = i + j;
    util::PrintMatWindow(M, m, n, {0,m-1}, {0, n-1});

    double* C;
    double* Z;
    int rank;
    int maxdim = 2;
    dInterpolative_PivotedQR(M, m, n, maxdim, C, Z, rank);
    

    delete[] M;
}

TEST(TTSVD_TEST, Way3_TTSVD_dense1)
{   
    tblis::tensor<double> T({2, 4, 3});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                T(i,j,k) = i + j + k;    

    auto factors = TT_SVD_dense(T, 2, 1E-5);
    auto tensor = util::TT_Contraction_dense(factors);    

    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k) 
                max_error = std::max(std::abs(T(i,j,k) - tensor(i,j,k)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}

TEST(TTSVD_TEST, Way3_TTSVD_dense2)
{   
    tblis::tensor<double> T({5, 3, 5, 6});
    // A initialization
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l)
                   T(i,j,k,l) = i * j + k - l;    

    auto factors = TT_SVD_dense(T, 10, 1E-10);
    auto tensor = util::TT_Contraction_dense(factors);
        
    // Find the maximum error between output C and the correct answer
    double max_error = 0.0;
    for (int i=0; i<2; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<3; ++k)
                for (int l=0; l<5; ++l) 
                    max_error = std::max(std::abs(T(i,j,k,l) - tensor(i,j,k,l)), max_error);                 
    EXPECT_NEAR(max_error,0,1E-10);
}