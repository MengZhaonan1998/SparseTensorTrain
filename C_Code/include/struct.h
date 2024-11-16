#ifndef STRUCT_H
#define STRUCT_H

namespace decompRes
{

template<class T>
struct PrrlduRes {
    T* L;
    T* d;
    T* U;
    size_t* row_perm_inv;
    size_t* col_perm_inv;
    T inf_error;
};

}

namespace sparseRes
{

template<class T>
struct MatrixCOO {
    T* data;
    size_t* colIdx;
    size_t* rowIdx;
    size_t nnz;
    size_t row;
    size_t col;

    // Constructor
    MatrixCOO(size_t nnz_, size_t row_, size_t col_) : nnz(nnz_), row(row_), col(col_) {
        // Allocate memory for data, colIdx, rowIdx
        data = new T[nnz];
        colIdx = new size_t[nnz];
        rowIdx = new size_t[nnz];
        std::cout << "MatrixCOO constructed with nnz: " << nnz << ", rows: " << row << ", cols: " << col << std::endl;
    }

    // Destructor
    ~MatrixCOO() {
        delete[] data;
        delete[] colIdx;
        delete[] rowIdx;
    }
};

template<class T>
struct TensorCOO {
    // TODO...
};

}

#endif