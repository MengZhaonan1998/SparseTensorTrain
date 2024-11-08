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

TEST(LapackeTEST, QR_3by5)
{
    // Define a 3x3 matrix A (row-major order)
    int m = 3, n = 5;
    double A[15] = {12, -51, 4, 23, 26,
                   6, 167, -68, -43, -9,
                   -4, 24, -41, 3, 98};
                   
    double A_orig[15];
    std::copy(A, A + 15, A_orig);  // Make a copy of A for verification
    
    double* Q = new double[m * m];
    double* R = new double[m * n]{0.0};
    int* jpvt = new int[n];       // Pivot indices

    dPivotedQR(m, n, A, Q, R, jpvt);

    // Step 4: Verify that QR - A is near zero
    double error = verifyQR(m, n, Q, R, A_orig, jpvt);
    EXPECT_NEAR(error, 0.0, 1E-10);

    delete[] Q;
    delete[] R;
    delete[] jpvt;
}

TEST(LapackeTEST, QR_5by3)
{
    // Define a 3x3 matrix A (row-major order)
    int m = 5, n = 3;
    double A[15] = {12, -51, 4, 23, 26,
                   6, 167, -68, -43, -9,
                   -4, 24, -41, 3, 98};
                   
    double A_orig[15];
    std::copy(A, A + 15, A_orig);  // Make a copy of A for verification
    
    double* Q = new double[m * m];
    double* R = new double[m * n]{0.0};
    int* jpvt = new int[n];       // Pivot indices

    dPivotedQR(m, n, A, Q, R, jpvt);

    // Step 4: Verify that QR - A is near zero
    double error = verifyQR(m, n, Q, R, A_orig, jpvt);
    EXPECT_NEAR(error, 0.0, 1E-10);

    delete[] Q;
    delete[] R;
    delete[] jpvt;
}