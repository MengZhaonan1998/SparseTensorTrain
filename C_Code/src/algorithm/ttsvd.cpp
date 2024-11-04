#include "header.h"

void fSVD(float* A, int m, int n, float* S, float* U, float* VT) {
    int lda = n;      // Leading dimension of A
    int ldu = m;      // Leading dimension of U
    int ldvt = n;     // Leading dimension of VT
    int info;
    // Query for optimal workspace size
    float work_size;
    info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, &work_size, -1);
    if (info != 0) throw std::runtime_error("SVD workspace query failed.");
    // Allocate optimal workspace
    int lwork = static_cast<int>(work_size);
    std::vector<float> work(lwork);
    // Call the LAPACKE SVD function with the allocated workspace
    info = LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work.data(), lwork);
    if (info > 0) throw std::runtime_error("SVD computation did not converge.");
}

void dSVD(double* A, int m, int n, double* S, double* U, double* VT) {
    int lda = n;      // Leading dimension of A
    int ldu = m;      // Leading dimension of U
    int ldvt = n;     // Leading dimension of VT
    int info;
    // Query for optimal workspace size
    double work_size;
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, &work_size, -1);
    if (info != 0) throw std::runtime_error("SVD workspace query failed.");
    // Allocate optimal workspace
    int lwork = static_cast<int>(work_size);
    std::vector<double> work(lwork);
    // Call the LAPACKE SVD function with the allocated workspace
    info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work.data(), lwork);
    if (info > 0) throw std::runtime_error("SVD computation did not converge.");
}


std::vector<tblis::tensor<double>> TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps)
{    
    // Initial setting
    auto shape = tensor.lengths(); // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();        // Get the number of dimension d
    double delta = (eps / std::sqrt(dim - 1)) * util::Norm(tensor, 2);  // Truncation parameter
    auto W = tensor;               // Copy tensor to W 
    auto nbar = util::GetSize(W);  // Total size of W
    int r = 1;                     // Rank r
    std::vector<tblis::tensor<double>> ttList;  // List storing TT factors
    bool verbose = 0;
    // TT-SVD iteration. Iterate from d-1 to 1
    for (int i = dim-1; i>0; i--) {
        int row = nbar / r / shape[i];       // Reshape row
        int col = r * shape[i];              // Reshape column
        // Initialize SVD factors
        double* U = new double[row * row];   
        double* Vh = new double[col * col];
        double* S = new double[std::min(row, col)];
        // Dense double SVD (Lapacke)
        dSVD(W.data(), row, col, S, U, Vh);
        // Cutoff selection
        double s = 0;
        int j = std::min(row, col);
        while (s <= delta * delta) {
            j -= 1;
            s += S[j] * S[j];
        }
        j += 1;
        int ri = std::min(j, r_max);
        // Print the low rank approximation error
        if (verbose) {
            std::cout << "Iter" << i << std::endl;
            // TODO...
        }
        // Form a new TT factor
        tblis::tensor<double> factor({ri, shape[i], r});
        if (i == dim - 1) factor.resize({ri, shape[i]});
        std::copy(Vh, Vh + ri * shape[i] * r, factor.data());
        nbar = nbar * ri / shape[i] / r;  // New total size of W
        r = ri;
        // To be polished/modified 
        W.resize({row, ri});
        for (int i = 0; i<row; ++i)
            for (int j=0; j<ri; ++j) {
                W(i,j) = U[i * row + j] * S[j];
            }
        // Append the factor to the ttList
        ttList.push_back(factor);    
        delete[] U;
        delete[] Vh;
        delete[] S;
    }
    // Append the last factor 
    ttList.push_back(W);
    std::reverse(ttList.begin(), ttList.end());
    return ttList;
}