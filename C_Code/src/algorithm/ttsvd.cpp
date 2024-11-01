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

double FrobNorm(tblis::tensor<double> tensor)
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

int GetSize(tblis::tensor<double> tensor)
{
    auto shape = tensor.lengths(); 
    int dim = shape.size();       
    int len = 1;
    for (int i=0; i<dim; ++i) len *= shape[i];
    return len;
}

void TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps)
{    
    // Initial setting
    auto shape = tensor.lengths(); // Get the shape of the input tensor: [n1, n2, ..., nd]
    int dim = shape.size();        // Get the number of dimension d
    double delta = (eps / std::sqrt(dim - 1)) * FrobNorm(tensor);  // Truncation parameter
    auto W = tensor;        // Copy tensor to W 
    auto nbar = GetSize(W); // Total size of W
    int r = 1;        // Rank r
    std::vector<tblis::tensor<double>> ttList;  // List storing TT factors
    bool verbose = 1;
    
    // TT-SVD iteration. Iterate from d-1 to 1
    for (int i = dim-1; i>0; i--)
    {
        auto M = W.data();
        int row = nbar / r / shape[i];
        int col = r * shape[i];
        double* U = new double[row * row];
        double* Vh = new double[col * col];
        double* S = new double[std::min(row, col)];

        dSVD(M, row, col, S, U, Vh);
        double s = 0;
        int j = std::min(row, col);
        while (s <= delta * delta)
        {
            j -= 1;
            s += S[j] * S[j];
        }
        j += 1;
        int ri = std::min(j, r_max);

        // To be continued...

        delete[] U;
        delete[] Vh;
        delete[] S;
    }
    

    //std::cout << shape << std::endl;
    return;
}