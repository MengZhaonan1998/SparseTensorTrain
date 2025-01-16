#include <gtest/gtest.h>
#include "new/spmatrix.h"
#include "new/functions.h"
#include "new/util.h"

TEST(SparsePRRLDU, SparseMat4by5)
{
    // COO-format matrix
    size_t Nr = 4;
    size_t Nc = 5;
    COOMatrix_l2<double> M_(Nr, Nc, 20);
    M_.add_element(0, 1, 2.0);
    M_.add_element(0, 3, -3.0);
    M_.add_element(0, 4, 1.2);
    M_.add_element(1, 2, -0.5);
    M_.add_element(1, 4, 4.0);
    M_.add_element(2, 0, -1.5);
    M_.add_element(2, 3, 5.0);
    M_.add_element(3, 1, -9.9);
    M_.add_element(3, 2, 0.2);
    //util::PrintMatWindow(M_.todense(), 4, 5, {0,3},{0,4});
    
    // Sparse partial rank-revealing LDU decomposition
    bool isFullReturn = true;
    double cutoff = 1e-8;
    double spthres = 0.6;
    size_t maxdim = 5;
    auto lduResult = dSparse_PartialRRLDU_CPU(M_, cutoff, spthres, maxdim, isFullReturn);
    size_t rank = lduResult.rank;
    size_t output_rank = lduResult.output_rank;
    size_t* row_perm_inv = lduResult.row_perm_inv;
    size_t* col_perm_inv = lduResult.col_perm_inv;
    double* d = lduResult.d;

    if (lduResult.isSparseRes) {
        // Reconstruction
        // L * d * U = pivoted M
        COOMatrix_l2<double> reconM(Nr, Nc, 20);
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) {
                double val = lduResult.sparse_L.get(i, j) * d[j];
                lduResult.sparse_L.update(i, j, val);
            }
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k) {
                    double val = lduResult.sparse_L.get(i, k) * lduResult.sparse_U.get(k, j);
                    reconM.addUpdate(i, j, val);                 
                }
        }

        // Reverse permutation
        double max_error = 0.0;
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM.get(row_perm_inv[i], col_perm_inv[j]);
                max_error = std::max(max_error, std::abs(recover - M_.get(i, j)));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
    } else {
        // Reconstruction
        // L * d * U = pivoted M
        double* reconM = new double[Nr * Nc]{0.0};
        double* L = lduResult.dense_L;
        double* U = lduResult.dense_U;
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) 
                L[i * output_rank + j] = L[i * output_rank + j] * d[j];
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k)
                    reconM[i * Nc + j] += L[i * output_rank + k] * U[k * Nc + j];
        }
        
        // Reverse permutation
        double max_error = 0.0;
        double* M_full = M_.todense();
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM[row_perm_inv[i] * Nc + col_perm_inv[j]];
                max_error = std::max(max_error, std::abs(recover - M_full[i * Nc + j]));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
        
        // Memory release
        delete[] M_full;
        delete[] reconM;
        lduResult.freeSpLduRes();
    }
}

TEST(SparsePRRLDU, SparseMat6by6)
{
    // COO-format matrix
    size_t Nr = 6;
    size_t Nc = 6;
    COOMatrix_l2<double> M_(6, 6, 36);
    M_.add_element(0, 1, 2.0);
    M_.add_element(0, 3, -3.0);
    M_.add_element(0, 4, 1.2);
    M_.add_element(0, 5, 9.2);
    M_.add_element(1, 2, -0.5);
    M_.add_element(1, 4, 4.0);
    M_.add_element(1, 5, 51.0);
    M_.add_element(2, 0, -1.5);
    M_.add_element(2, 3, 5.0);
    M_.add_element(3, 2, -99);
    M_.add_element(4, 2, 0.6);
    M_.add_element(5, 2, 2.4);
    //util::PrintMatWindow(M_.todense(), 4, 5, {0,3},{0,4});
    
    // Sparse partial rank-revealing LDU decomposition
    bool isFullReturn = true;
    double cutoff = 1e-8;
    double spthres = 0.36;
    size_t maxdim = 5;
    auto lduResult = dSparse_PartialRRLDU_CPU(M_, cutoff, spthres, maxdim, isFullReturn);
    size_t rank = lduResult.rank;
    size_t output_rank = lduResult.output_rank;
    size_t* row_perm_inv = lduResult.row_perm_inv;
    size_t* col_perm_inv = lduResult.col_perm_inv;
    double* d = lduResult.d;

    if (lduResult.isSparseRes) {
        // Reconstruction
        // L * d * U = pivoted M
        COOMatrix_l2<double> reconM(Nr, Nc, 20);
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) {
                double val = lduResult.sparse_L.get(i, j) * d[j];
                lduResult.sparse_L.update(i, j, val);
            }
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k) {
                    double val = lduResult.sparse_L.get(i, k) * lduResult.sparse_U.get(k, j);
                    reconM.addUpdate(i, j, val);                 
                }
        }
        
        // Reverse permutation
        double max_error = 0.0;
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM.get(row_perm_inv[i], col_perm_inv[j]);
                max_error = std::max(max_error, std::abs(recover - M_.get(i, j)));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
    } else {
        // Reconstruction
        // L * d * U = pivoted M
        double* reconM = new double[Nr * Nc]{0.0};
        double* L = lduResult.dense_L;
        double* U = lduResult.dense_U;
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) 
                L[i * output_rank + j] = L[i * output_rank + j] * d[j];
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k)
                    reconM[i * Nc + j] += L[i * output_rank + k] * U[k * Nc + j];
        }
        
        // Reverse permutation
        double max_error = 0.0;
        double* M_full = M_.todense();
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM[row_perm_inv[i] * Nc + col_perm_inv[j]];
                max_error = std::max(max_error, std::abs(recover - M_full[i * Nc + j]));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
        
        // Memory release
        delete[] M_full;
        delete[] reconM;
        lduResult.freeSpLduRes();
    }
}

TEST(SparsePRRLDU, SparseMatRandom)
{
    COOMatrix_l2<double> A(10, 6, 60);
    COOMatrix_l2<double> B(6, 8, 48);

    // Generate random entries
    double density = 0.3;   
    unsigned int seed = 42; 
    A.generate_random(density, seed, -10.0, 10.0);
    seed = 76;
    B.generate_random(density, seed, -10.0, 10.0);
    COOMatrix_l2<double> M_ = A.multiply(B);
    
    size_t Nr = M_.rows;
    size_t Nc = M_.cols;
    //util::PrintMatWindow(M_.todense(), Nr, Nc, {0,Nr-1},{0,Nc-1});

    // Sparse partial rank-revealing LDU decomposition
    bool isFullReturn = true;
    double cutoff = 1e-8;
    double spthres = 0.45;
    size_t maxdim = 8;
    auto lduResult = dSparse_PartialRRLDU_CPU(M_, cutoff, spthres, maxdim, isFullReturn);
    size_t rank = lduResult.rank;
    size_t output_rank = lduResult.output_rank;
    size_t* row_perm_inv = lduResult.row_perm_inv;
    size_t* col_perm_inv = lduResult.col_perm_inv;
    double* d = lduResult.d;

    if (lduResult.isSparseRes) {
        // Reconstruction
        // L * d * U = pivoted M
        COOMatrix_l2<double> reconM(Nr, Nc, 20);
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) {
                double val = lduResult.sparse_L.get(i, j) * d[j];
                lduResult.sparse_L.update(i, j, val);
            }
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k) {
                    double val = lduResult.sparse_L.get(i, k) * lduResult.sparse_U.get(k, j);
                    reconM.addUpdate(i, j, val);                 
                }
        }
        
        // Reverse permutation
        double max_error = 0.0;
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM.get(row_perm_inv[i], col_perm_inv[j]);
                max_error = std::max(max_error, std::abs(recover - M_.get(i, j)));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
    } else {
        // Reconstruction
        // L * d * U = pivoted M
        double* reconM = new double[Nr * Nc]{0.0};
        double* L = lduResult.dense_L;
        double* U = lduResult.dense_U;
        for (int i = 0; i < Nr; ++i) {
            for (int j = 0; j < output_rank; ++j) 
                L[i * output_rank + j] = L[i * output_rank + j] * d[j];
            for (int j = 0; j < Nc; ++j)
                for (int k = 0; k < output_rank; ++k)
                    reconM[i * Nc + j] += L[i * output_rank + k] * U[k * Nc + j];
        }
        
        // Reverse permutation
        double max_error = 0.0;
        double* M_full = M_.todense();
        for (int i = 0; i < Nr; ++i) 
            for (int j = 0; j < Nc; ++j) {
                double recover = reconM[row_perm_inv[i] * Nc + col_perm_inv[j]];
                max_error = std::max(max_error, std::abs(recover - M_full[i * Nc + j]));
            }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
        
        // Memory release
        delete[] M_full;
        delete[] reconM;
        lduResult.freeSpLduRes();
    }
}

TEST(SparseIDprrldu, SparseMat4by5)
{
    // COO-format matrix
    size_t Nr = 4;
    size_t Nc = 5;
    COOMatrix_l2<double> M_(Nr, Nc, Nr * Nc);
    M_.add_element(0, 1, 2.0);
    M_.add_element(0, 3, -3.0);
    M_.add_element(0, 4, 1.2);
    M_.add_element(1, 2, -0.5);
    M_.add_element(1, 4, 4.0);
    M_.add_element(2, 0, -1.5);
    M_.add_element(2, 3, 5.0);
    M_.add_element(3, 1, -9.9);
    M_.add_element(3, 2, 0.2);
     
    // Sparse interpolative decomposition based on partial rank-revealing LDU decomposition
    bool isFullReturn = true;
    double cutoff = 1e-8;
    double spthres = 0.45;
    size_t maxdim = 5;
    auto idResult = dSparse_Interpolative_CPU(M_, cutoff, spthres, maxdim);

    // Reconstruction of C/Z
    size_t output_rank = idResult.output_rank;
    auto C = M_.subcol(idResult.pivot_cols, output_rank);
    auto Z = dcoeffZRecon(idResult.interp_coeff, idResult.pivot_cols, output_rank, Nc);
    
    // Reconstruction M
    auto recon = C.multiply(Z);
    double max_error = 0.0;
    double* M_full = M_.todense();
    double* recon_full = recon.todense();
    for (int i = 0; i < Nr; ++i) 
        for (int j = 0; j < Nc; ++j) {
            max_error = std::max(max_error, std::abs(recon_full[i * Nc + j] - M_full[i * Nc + j]));
        }
    EXPECT_NEAR(max_error, 0.0, 1e-8);

    delete[] recon_full;
    delete[] M_full;
    idResult.freeSpInterpRes();
}

TEST(SparseIDprrldu, SparseMat6by6)
{
    // COO-format matrix
    size_t Nr = 6;
    size_t Nc = 6;
    COOMatrix_l2<double> M_(6, 6, 36);
    M_.add_element(0, 1, 2.0);
    M_.add_element(0, 3, -3.0);
    M_.add_element(0, 4, 1.2);
    M_.add_element(0, 5, 9.2);
    M_.add_element(1, 2, -0.5);
    M_.add_element(1, 4, 4.0);
    M_.add_element(1, 5, 51.0);
    M_.add_element(2, 0, -1.5);
    M_.add_element(2, 3, 5.0);
    M_.add_element(3, 2, -99);
    M_.add_element(4, 2, 0.6);
    M_.add_element(5, 2, 2.4);
    
    // Sparse interpolative decomposition based on partial rank-revealing LDU decomposition
    double cutoff = 1e-8;
    double spthres = 0.36;
    size_t maxdim = 6;
    auto idResult = dSparse_Interpolative_CPU(M_, cutoff, spthres, maxdim);

    // Reconstruction of C/Z
    size_t output_rank = idResult.output_rank;
    auto C = M_.subcol(idResult.pivot_cols, output_rank);
    auto Z = dcoeffZRecon(idResult.interp_coeff, idResult.pivot_cols, output_rank, Nc);
    
    // Reconstruction M
    auto recon = C.multiply(Z);
    double max_error = 0.0;
    double* M_full = M_.todense();
    double* recon_full = recon.todense();
    for (int i = 0; i < Nr; ++i) 
        for (int j = 0; j < Nc; ++j) {
            max_error = std::max(max_error, std::abs(recon_full[i * Nc + j] - M_full[i * Nc + j]));
        }
    EXPECT_NEAR(max_error, 0.0, 1e-8);

    delete[] recon_full;
    delete[] M_full;
    idResult.freeSpInterpRes();
}

TEST(SparseIDprrldu, SparseMatRandom)
{
    COOMatrix_l2<double> A(30, 25);
    COOMatrix_l2<double> B(25, 40);

    // Generate random entries
    double density = 0.1;   
    unsigned int seed = 42; 
    A.generate_random(density, seed, -10.0, 10.0);
    seed = 76;
    B.generate_random(density, seed, -10.0, 10.0);
    COOMatrix_l2<double> M_ = A.multiply(B);
    size_t Nr = M_.rows;
    size_t Nc = M_.cols;

    // Sparse partial rank-revealing LDU decomposition
    double cutoff = 1e-8;
    double spthres = 0.40;
    size_t maxdim = 50;
    auto idResult = dSparse_Interpolative_CPU(M_, cutoff, spthres, maxdim);

    // Reconstruction of C/Z
    size_t output_rank = idResult.output_rank;
    auto C = M_.subcol(idResult.pivot_cols, output_rank);
    auto Z = dcoeffZRecon(idResult.interp_coeff, idResult.pivot_cols, output_rank, Nc);
    
    // Reconstruction M
    auto recon = C.multiply(Z);
    double max_error = 0.0;
    double* M_full = M_.todense();
    double* recon_full = recon.todense();
    for (int i = 0; i < Nr; ++i) 
        for (int j = 0; j < Nc; ++j) {
            max_error = std::max(max_error, std::abs(recon_full[i * Nc + j] - M_full[i * Nc + j]));
        }
    EXPECT_NEAR(max_error, 0.0, 1e-8);

    delete[] recon_full;
    delete[] M_full;
    idResult.freeSpInterpRes();
}