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

TEST(SparseTensor_TEST, COO_Sparse2Dense)
{
// Create tensors of different orders
    COOTensor<double, 3> tensor3d(10, 3, 4, 5);       // 3rd order tensor
    COOTensor<double, 5> tensor5d(10, 3, 2, 4, 3, 1); // 5th order tensor
   
    tensor3d.add_element(1.0, 0, 0, 0);
    tensor3d.add_element(2.0, 1, 1, 1);
    tensor3d.add_element(3.0, 2, 3, 2);
    tensor3d.add_element(4.0, 0, 2, 3);
    auto tensor3d_full = tensor3d.to_dense(); 
    EXPECT_NEAR(1.0, tensor3d_full(0,0,0), 1E-10);
    EXPECT_NEAR(2.0, tensor3d_full(1,1,1), 1E-10);
    EXPECT_NEAR(3.0, tensor3d_full(2,3,2), 1E-10);
    EXPECT_NEAR(4.0, tensor3d_full(0,2,3), 1E-10);

    tensor5d.add_element(1.0, 0, 0, 0, 0, 0);
    tensor5d.add_element(2.0, 1, 1, 3, 1, 0);
    tensor5d.add_element(3.0, 2, 1, 2, 0, 0);
    tensor5d.add_element(4.0, 2, 0, 3, 2, 0);
    tensor5d.add_element(5.0, 1, 1, 1, 0, 0);
    auto tensor5d_full = tensor5d.to_dense(); 
    EXPECT_NEAR(1.0, tensor5d_full(0,0,0,0,0), 1E-10);
    EXPECT_NEAR(2.0, tensor5d_full(1,1,3,1,0), 1E-10);
    EXPECT_NEAR(3.0, tensor5d_full(2,1,2,0,0), 1E-10);
    EXPECT_NEAR(4.0, tensor5d_full(2,0,3,2,0), 1E-10);
    EXPECT_NEAR(5.0, tensor5d_full(1,1,1,0,0), 1E-10);
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
    A.add_element(1.0, 0, 1, 2);
    A.add_element(2.0, 1, 2, 3);
    A.add_element(5.0, 3, 2, 5);
    A.add_element(-0.3,2, 4, 1);
    B.add_element(3.0, 2, 0, 1);
    B.add_element(4.0, 3, 1, 2);
    B.add_element(-1.2,5, 6, 2);
    
    // Contract A's third dimension (index 2) with B's first dimension (index 0)
    auto C = A.contract(B, 2, 0);  // Result will be 4x5x7 tensor
    
    // Check the correctness
    EXPECT_EQ(C.nnz(), 3);   // Number of non-zeros
    EXPECT_NEAR(3.0, C.get(0,1,0,1), 1E-10);
    EXPECT_NEAR(8.0, C.get(1,2,1,2), 1E-10);
    EXPECT_NEAR(-6.0,C.get(3,2,6,2), 1E-10);
}

TEST(SparseTensor_TEST, COO_TensorTrainContraction1)
{
    // Create TT-cores with dimensions:
    // G1(1,n1,r1), G2(r1,n2,r2), G3(r2,n3,1)
    COOTensor<float, 2> G1(100, 4, 3);    // rank0=1, n1=4, rank1=3
    COOTensor<float, 3> G2(100, 3, 5, 2); // rank1=3, n2=5, rank2=2
    COOTensor<float, 2> G3(100, 2, 6);    // rank2=2, n3=6, rank3=1
    
    // Fill cores with some values
    G1.add_element(1.0f, 0, 0);
    G1.add_element(2.0f, 1, 1);
    
    G2.add_element(3.0f, 0, 0, 0);
    G2.add_element(4.0f, 1, 1, 0);
    
    G3.add_element(5.0f, 0, 0);
    G3.add_element(6.0f, 0, 1);
    
    // Contract the entire train
    // Result will be a tensor of shape (4,5,6)
    auto result = SparseTTtoTensor3<float>(G1, G2, G3);       

    // Check the correctness
    EXPECT_EQ(result.nnz(), 4);   // Number of non-zeros
    EXPECT_NEAR(15.0f, result.get(0,0,0), 1E-8);
    EXPECT_NEAR(18.0f, result.get(0,0,1), 1E-8);
    EXPECT_NEAR(40.0f, result.get(1,1,0), 1E-8);
    EXPECT_NEAR(48.0f, result.get(1,1,1), 1E-8);
}

TEST(SparseTensor_TEST, COO_TensorTrainContraction2)
{
    // Create TT-cores with dimensions:
    COOTensor<double, 2> G1(100, 2, 5);    
    COOTensor<double, 3> G2(100, 5, 4, 3); 
    COOTensor<double, 3> G3(100, 3, 6, 2);
    COOTensor<double, 2> G4(100, 2, 10); 
    
    // Fill cores with some values
    G1.add_element(-1.0,1, 4);
    G1.add_element(2.0, 1, 1);
    G1.add_element(4.2, 1, 2);

    G2.add_element(3.0, 0, 3, 0);
    G2.add_element(4.0, 4, 1, 2);
    
    G3.add_element(5.4, 0, 0, 0);
    G3.add_element(-2.0,0, 4, 1);
    G3.add_element(3.0, 1, 1, 0);
    G3.add_element(1.1, 2, 3, 1);

    G4.add_element(-4.7,0, 4);
    G4.add_element(8.6, 1, 8);
    G4.add_element(10.0,1, 7);

    // Contract the entire train
    auto result = SparseTTtoTensor4<double>(G1, G2, G3, G4);       

    // Check the correctness
    EXPECT_EQ(result.nnz(), 2);   // Number of non-zeros
    EXPECT_NEAR(-37.84,result.get(1,1,3,8), 1E-10);
    EXPECT_NEAR(-44.0, result.get(1,1,3,7), 1E-10);
}

TEST(SparseTensor_TEST, COO_DataIO)
{
    COOTensor<double, 3> tensor1(10, 2, 3, 4);
    tensor1.add_element(3.2, 0, 0, 0);
    tensor1.add_element(1.3, 0, 1, 1);
    tensor1.add_element(-9.7,0, 1, 2);
    tensor1.add_element(-8.4,1, 2, 3);
    tensor1.write_to_file("tensor.txt");

    COOTensor<double, 3> tensor2(10, 2, 3, 4);   
    tensor2.read_from_file("tensor.txt");     
    
    EXPECT_EQ(tensor2.nnz(), 4);   // Number of non-zeros
    EXPECT_NEAR(3.2, tensor2.get(0,0,0), 1E-10);
    EXPECT_NEAR(1.3, tensor2.get(0,1,1), 1E-10);
    EXPECT_NEAR(-9.7,tensor2.get(0,1,2), 1E-10);
    EXPECT_NEAR(-8.4,tensor2.get(1,2,3), 1E-10);

    auto full_tensor = tensor2.to_dense();
    EXPECT_NEAR(3.2, full_tensor(0,0,0), 1E-10);
    EXPECT_NEAR(1.3, full_tensor(0,1,1), 1E-10);
    EXPECT_NEAR(-9.7,full_tensor(0,1,2), 1E-10);
    EXPECT_NEAR(-8.4,full_tensor(1,2,3), 1E-10);

    if (std::remove("tensor.txt") != 0) {
        throw std::runtime_error("Error deleting file: tensor.txt");
    }
}