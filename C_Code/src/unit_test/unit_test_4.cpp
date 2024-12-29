#include <gtest/gtest.h>
#include "new/spmatrix.h"
#include "new/sptensor.h"

TEST(SparseMat_TEST, COO_MatStruct)
{
    // Create a 4x4 sparse matrix
    COOMatrix_l2<double> matrix(4, 4);

    // Add some non-zero elements
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(1, 1, 2.0);
    matrix.add_element(2, 3, 3.0);
    matrix.add_element(3, 2, 4.0);

    // Sort the elements
    matrix.sort();

    // Print the matrix
    //matrix.print();

    // Get a specific element
    //std::cout << "Value at (2,3): " << matrix.get(2, 3) << "\n";
    //std::cout << "Value at (1,2): " << matrix.get(1, 2) << "\n";

    // Test copy constructor
    COOMatrix_l2<double> matrix2 = matrix;
    //std::cout << "\nCopied matrix:\n";
    //matrix2.print();
}

TEST(SparseTensor_TEST, COO_TensorStruct)
{
// Create tensors of different orders
    COOTensor<double, 3> tensor3d(10, 3, 3, 3);  // 3rd order tensor
    COOTensor<double, 4> tensor4d(10, 2, 2, 2, 2);  // 4th order tensor
    COOTensor<double, 5> tensor5d(10, 2, 2, 2, 2, 2);  // 5th order tensor

    // Add elements to 3D tensor
    tensor3d.add_element(1.0, 0, 0, 0);
    tensor3d.add_element(2.0, 1, 1, 1);
    //tensor3d.print();

    // Add elements to 4D tensor
    tensor4d.add_element(1.0, 0, 0, 0, 0);
    tensor4d.add_element(2.0, 1, 1, 1, 1);
    //tensor4d.print();

    // Add elements to 5D tensor
    tensor5d.add_element(1.0, 0, 0, 0, 0, 0);
    tensor5d.add_element(2.0, 1, 1, 1, 1, 1);
    //tensor5d.print();
}

TEST(SparseTensor_TEST, COO_TensorContraction1)
{ 
    COOTensor<float, 2> A(10, 4, 6);  // 4x6 tensor
    COOTensor<float, 2> B(10, 6, 7);  // 6x7 tensor
    
    // Add some elements
    A.add_element(1.0f, 0, 1);
    A.add_element(2.0f, 1, 2);
    A.add_element(-2.5f,3, 3);
    B.add_element(3.0f, 2, 0);
    B.add_element(4.0f, 3, 1);

    // Contract A's second dimension (index 1) with B's first dimension (index 0)
    auto C = A.contract(B, 1, 0);  // Result will be 4x7 tensor
    
    // Check the correctness
    EXPECT_EQ(C.nnz(), 2);   // Number of non-zeros
    EXPECT_NEAR(6.0f, C.get(1,0), 1E-8);
    EXPECT_NEAR(-10.0f, C.get(3,1), 1E-8);
}

TEST(SparseTensor_TEST, COO_TensorContraction2)
{
    // Create tensors
    COOTensor<double, 3> A(100, 4, 5, 6);  // 4x5x6 tensor
    COOTensor<double, 3> B(100, 6, 7, 3);  // 6x7x3 tensor
    
    // Add some elements
    A.add_element(1.0f, 0, 1, 2);
    A.add_element(2.0f, 1, 2, 3);
    A.add_element(5.0f, 3, 2, 5);
    A.add_element(-0.3f,2, 4, 1);
    B.add_element(3.0f, 2, 0, 1);
    B.add_element(4.0f, 3, 1, 2);
    B.add_element(-1.2, 5, 6, 2);
    
    // Contract A's third dimension (index 2) with B's first dimension (index 0)
    auto C = A.contract(B, 2, 0);  // Result will be 4x5x7 tensor
    
    // Check the correctness
    EXPECT_EQ(C.nnz(), 3);   // Number of non-zeros
    EXPECT_NEAR(3.0f, C.get(0,1,0,1), 1E-10);
    EXPECT_NEAR(8.0f, C.get(1,2,1,2), 1E-10);
    EXPECT_NEAR(-6.0f,C.get(3,2,6,2), 1E-10);
}