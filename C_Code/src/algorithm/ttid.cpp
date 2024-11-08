#include "header.h"

void dInterpolative_QR(double* M, int m, int n, int maxdim, double* C, double* Z)
{
    int k = std::min(m, n);
    k = maxdim < k ? maxdim : k;
    
    double* Q = new double[m * m]{0.0};
    double* R = new double[m * n]{0.0};
    int* jpvt = new int[n];
    dPivotedQR(m, n, M, Q, R, jpvt);

    std::cout << "Q" << std::endl;
    util::PrintMatWindow(Q, m, m, {0,m-1}, {0,m-1});
    std::cout << "R" << std::endl;
    util::PrintMatWindow(R, m, n, {0,m-1}, {0,n-1});
    


    delete[] Q;
    delete[] R;
    delete[] jpvt;
    return;        
}

void TT_ID()
{

}