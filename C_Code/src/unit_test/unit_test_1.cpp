#include <gtest/gtest.h>
#include "header.h"

TEST(LapackeTEST, SVD_2by2)
{
    // Sample 2x2 matrix
    float A[4] = {1.0, 2.0, 3.0, 4.0};
    float S[2], U[4], VT[4];
    int m = 2, n = 2;   
    // Call the SVD function you implemented in ttsvd.cpp
    fSVD(A, m, n, S, U, VT);
    // Assert that the singular values are as expected
    EXPECT_NEAR(S[0], 5.46499, 1E-4); // Expected first singular value
    EXPECT_NEAR(S[1], 0.36596, 1E-4); // Expected second singular value
}

TEST(LapackeTEST, SVD_3by3) 
{
    double A[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double S[3];  // Singular values
    double U[9];  // Left singular vectors
    double VT[9]; // Right singular vectors transposed
    int m = 3, n = 3;
    // Call the SVD function you implemented in ttsvd.cpp
    dSVD(A, m, n, S, U, VT);
    // Assert the expected singular values within a tolerance
    EXPECT_NEAR(S[0], 16.8481, 1E-4);
    EXPECT_NEAR(S[1], 1.0684, 1E-4);
    EXPECT_NEAR(S[2], 0.0, 1E-4);
    // Optional: Further checks on U and VT matrices can be added if needed.
}

