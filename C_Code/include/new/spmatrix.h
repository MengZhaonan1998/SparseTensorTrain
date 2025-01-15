// spmatrix.h - Sparse matrix toolkit
#ifndef SPMATRIX_H
#define SPMATRIX_H

#include "core.h"

template<typename T>
struct COOMatrix_l1 {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t capacity;
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
    COOMatrix_l2(size_t num_rows, size_t num_cols, size_t initial_capacity = 100) 
        : rows(num_rows), cols(num_cols), capacity(initial_capacity), nnz_count(0) {
        row_indices = new size_t[capacity];
        col_indices = new size_t[capacity];
        values = new T[capacity];
    }

    // Destructor
    ~COOMatrix_l2() {
        if (row_indices != nullptr)
            delete[] row_indices;
        if (col_indices != nullptr)
            delete[] col_indices;
        if (values != nullptr)
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

    // Update the value at a specific position
    void update(size_t row, size_t col, T val) {
        for (size_t i = 0; i < nnz_count; ++i) {
            if (row_indices[i] == row && col_indices[i] == col) {
                values[i] = val; 
                return;                
            }
        }
        add_element(row, col, val);
    }

    // Update the value by +=
    void addUpdate(size_t row, size_t col, T val) {
        for (size_t i = 0; i < nnz_count; ++i) {
            if (row_indices[i] == row && col_indices[i] == col) {
                values[i] += val; 
                return;                
            }
        }
        add_element(row, col, val);
    }

    // Full format
    T* todense() {
        T* full = new T[rows * cols]{0};
        for (size_t i = 0; i < nnz_count; ++i) {
            full[row_indices[i] * cols + col_indices[i]] = values[i];
        }
        return full;
    }

    // In-place reshape
    void reshape(size_t new_row, size_t new_col) {
        if (new_row * new_col != rows * cols) {
            throw std::runtime_error("New shape does not match with the original dimension!");
        }
        // Reshape
        size_t idx;
        for (size_t i = 0; i < nnz_count; ++i) {
            idx = row_indices[i] * cols + col_indices[i];
            row_indices[i] = idx / new_col;
            col_indices[i] = idx % new_col;
        }
        return;    
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

    void explicit_destroy() {
        if (row_indices != nullptr) {
            delete[] row_indices;
            row_indices = nullptr;
        }
        if (col_indices != nullptr) {
            delete[] col_indices;
            col_indices = nullptr;
        }
        if (values != nullptr) {
            delete[] values;
            values = nullptr;
        }
        nnz_count = 0;
    }

    COOMatrix_l2<T> subcol(const size_t* pivot_cols, size_t rank) {
        COOMatrix_l2<T> result(rows, rank);
        
        // Select columns from pivot column array
        for (size_t i = 0; i < nnz_count; ++i) {
            size_t col_idx = col_indices[i];
            for (size_t j = 0; j < rank; ++j) {
                size_t pcol_idx = pivot_cols[j];
                if (pcol_idx == col_idx) {
                    result.add_element(row_indices[i], j, values[i]);
                }
            }    
        }
        
        return result;
    }

    COOMatrix_l2<T> multiply(const COOMatrix_l2<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        // Initialize result matrix
        COOMatrix_l2<T> result(rows, other.cols);
        
        // Create a map to accumulate results for each position
        std::map<std::pair<size_t, size_t>, T> temp_results;
        
        // For each non-zero element in the first matrix
        for (size_t i = 0; i < nnz_count; ++i) {
            size_t row_a = row_indices[i];
            T val_a = values[i];
            
            // For each non-zero element in the second matrix
            for (size_t j = 0; j < other.nnz_count; ++j) {
                // Only multiply if the column of first matches row of second
                if (col_indices[i] == other.row_indices[j]) {
                    size_t col_b = other.col_indices[j];
                    T val_b = other.values[j];
                    
                    // Accumulate the product
                    temp_results[{row_a, col_b}] += val_a * val_b;
                }
            }
        }
        
        // Convert accumulated results to COO format
        for (const auto& entry : temp_results) {
            if (entry.second != T(0)) {  // Only store non-zero results
                result.add_element(entry.first.first, entry.first.second, entry.second);
            }
        }
        
        // Sort the result for better access patterns
        result.sort();
        return result;
    }

    // Generate random non-zero entries
    void generate_random(double density, unsigned int seed, T min_val = T(1), T max_val = T(100)) {
        if (density < 0.0 || density > 1.0) {
            throw std::invalid_argument("Density must be between 0 and 1");
        }

        // Calculate number of non-zero elements based on density
        size_t total_elements = rows * cols;
        size_t target_nnz = static_cast<size_t>(density * total_elements);
        
        // Clear existing data
        explicit_destroy();
        
        // Initialize with new capacity
        capacity = target_nnz;
        row_indices = new size_t[capacity];
        col_indices = new size_t[capacity];
        values = new T[capacity];
        
        // Set up random number generators
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> row_dist(0, rows - 1);
        std::uniform_int_distribution<size_t> col_dist(0, cols - 1);
        
        // For floating point values
        std::uniform_real_distribution<double> val_dist(
            static_cast<double>(min_val), 
            static_cast<double>(max_val)
        );
        
        // Use set to ensure unique positions
        std::set<std::pair<size_t, size_t>> positions;
        
        // Generate unique random positions
        while (positions.size() < target_nnz) {
            size_t row = row_dist(gen);
            size_t col = col_dist(gen);
            positions.insert({row, col});
        }
        
        // Fill the matrix with random values at these positions
        nnz_count = 0;
        for (const auto& pos : positions) {
            T value;
            if constexpr (std::is_integral<T>::value) {
                // For integer types
                value = static_cast<T>(std::round(val_dist(gen)));
            } else {
                // For floating point types
                value = static_cast<T>(val_dist(gen));
            }
            
            row_indices[nnz_count] = pos.first;
            col_indices[nnz_count] = pos.second;
            values[nnz_count] = value;
            nnz_count++;
        }
        
        // Sort the entries
        sort();
    }
};

#endif