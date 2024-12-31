// spmatrix.h - Sparse matrix toolkit
#ifndef SPMATRIX_H
#define SPMATRIX_H

#include "core.h"

template<typename T>
struct COOMatrix_l1 {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_indices;
    size_t* col_indices;
    T* values;

    // Default constructor
    COOMatrix_l1() : rows(0), cols(0), nnz(0), 
                     row_indices(nullptr), 
                     col_indices(nullptr), 
                     values(nullptr) {}

    // Default destructor
    ~COOMatrix_l1() {
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
    }
};

template<typename T>
struct COOMatrix_l2 {
    size_t rows;  // Number of rows in the full matrix
    size_t cols;  // Number of columns in the full matrix
    size_t capacity;   // Maximum capacity for non-zero elements  
    size_t nnz_count;  // Current number of non-zero elements 
    
    // Arrays to store the non-zero elements and their positions
    size_t* row_indices;    // Row indices of non-zero elements
    size_t* col_indices;    // Column indices of non-zero elements
    T* values;              // Values of non-zero elements

    // Constructor
    COOMatrix_l2(size_t num_rows, size_t num_cols, size_t initial_capacity = 10) 
        : rows(num_rows), cols(num_cols), capacity(initial_capacity), nnz_count(0) {
        row_indices = new size_t[capacity];
        col_indices = new size_t[capacity];
        values = new T[capacity];
    }

    // Destructor
    ~COOMatrix_l2() {
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
    }

    // Copy constructor
    COOMatrix_l2(const COOMatrix_l2& other) 
        : rows(other.rows), cols(other.cols), 
          capacity(other.capacity), nnz_count(other.nnz_count) {
        row_indices = new size_t[capacity];
        col_indices = new size_t[capacity];
        values = new T[capacity];
        
        std::memcpy(row_indices, other.row_indices, nnz_count * sizeof(size_t));
        std::memcpy(col_indices, other.col_indices, nnz_count * sizeof(size_t));
        std::memcpy(values, other.values, nnz_count * sizeof(T));
    }

    // Assignment operator
    COOMatrix_l2& operator=(const COOMatrix_l2& other) {
        if (this != &other) {
            // Free existing resources
            delete[] row_indices;
            delete[] col_indices;
            delete[] values;
            
            // Copy new data
            rows = other.rows;
            cols = other.cols;
            capacity = other.capacity;
            nnz_count = other.nnz_count;
            
            row_indices = new size_t[capacity];
            col_indices = new size_t[capacity];
            values = new T[capacity];
            
            std::memcpy(row_indices, other.row_indices, nnz_count * sizeof(size_t));
            std::memcpy(col_indices, other.col_indices, nnz_count * sizeof(size_t));
            std::memcpy(values, other.values, nnz_count * sizeof(T));
        }
        return *this;
    }

    // Resize arrays when capacity is reached
    void resize(size_t new_capacity) {
        size_t* new_row_indices = new size_t[new_capacity];
        size_t* new_col_indices = new size_t[new_capacity];
        T* new_values = new T[new_capacity];
        
        std::memcpy(new_row_indices, row_indices, nnz_count * sizeof(size_t));
        std::memcpy(new_col_indices, col_indices, nnz_count * sizeof(size_t));
        std::memcpy(new_values, values, nnz_count * sizeof(T));
        
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
        
        row_indices = new_row_indices;
        col_indices = new_col_indices;
        values = new_values;
        capacity = new_capacity;
    }

    // Add a non-zero element to the matrix
    void add_element(size_t row, size_t col, T value) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (value != T(0)) {  // Only store non-zero values
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            row_indices[nnz_count] = row;
            col_indices[nnz_count] = col;
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    // Get the value at a specific position
    T get(size_t row, size_t col) const {
        for (size_t i = 0; i < nnz_count; ++i) {
            if (row_indices[i] == row && col_indices[i] == col) {
                return values[i];
            }
        }
        return T(0);  // Return zero if element not found
    }

    // Get number of non-zero elements
    size_t nnz() const {
        return nnz_count;
    }

    // Sort elements by row and column indices
    void sort() {
        size_t* indices = new size_t[nnz_count];
        for (size_t i = 0; i < nnz_count; ++i) {
            indices[i] = i;
        }

        std::sort(indices, indices + nnz_count,
            [this](size_t i1, size_t i2) {
                if (row_indices[i1] != row_indices[i2])
                    return row_indices[i1] < row_indices[i2];
                return col_indices[i1] < col_indices[i2];
            });

        // Create temporary arrays for sorting
        size_t* new_row_indices = new size_t[capacity];
        size_t* new_col_indices = new size_t[capacity];
        T* new_values = new T[capacity];

        for (size_t i = 0; i < nnz_count; ++i) {
            new_row_indices[i] = row_indices[indices[i]];
            new_col_indices[i] = col_indices[indices[i]];
            new_values[i] = values[indices[i]];
        }

        // Swap pointers
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
        delete[] indices;

        row_indices = new_row_indices;
        col_indices = new_col_indices;
        values = new_values;
    }

    // Print the matrix in COO format
    void print() const {
        std::cout << "COO Matrix (" << rows << " x " << cols << "), "
                  << nnz_count << " non-zero elements:\n";
        for (size_t i = 0; i < nnz_count; ++i) {
            std::cout << "(" << row_indices[i] << ", " << col_indices[i] 
                      << ") = " << values[i] << "\n";
        }
    }
};

#endif