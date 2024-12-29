#include "core.h"
#include "external.h"

// Helper for index sequence
template<size_t... Is>
struct IndexSequence {};

template<size_t N, size_t... Is>
struct MakeIndexSequence : MakeIndexSequence<N-1, N-1, Is...> {};

template<size_t... Is>
struct MakeIndexSequence<0, Is...> {
    using type = IndexSequence<Is...>;
};

template<typename T, size_t Order>
class COOTensor {
private:
    // Array to store dimensions
    std::array<size_t, Order> dimensions;

    // Maximum capacity for non-zero elements
    size_t capacity;
    
    // Current number of non-zero elements
    size_t nnz_count;
    
    // Arrays to store indices and values
    std::array<size_t*, Order> indices;  // Array of pointers to index arrays
    T* values;                           // Values of non-zero elements

public:
    // Helper function to check if indices are within bounds
    template<size_t... Is>
    bool check_bounds(const std::array<size_t, Order>& idx, IndexSequence<Is...>) const {
        return ((idx[Is] < dimensions[Is]) && ...);
    }

    // Helper function for lexicographic comparison
    bool compare_indices(size_t i1, size_t i2) const {
        for (size_t dim = 0; dim < Order; ++dim) {
            if (indices[dim][i1] != indices[dim][i2]) {
                return indices[dim][i1] < indices[dim][i2];
            }
        }
        return false;
    }
    
    // Helper struct for contraction specification
    struct ContractionDims {
        size_t this_dim;   // Dimension from this tensor
        size_t other_dim;  // Dimension from other tensor
        ContractionDims(size_t t, size_t o) : this_dim(t), other_dim(o) {}
    };

    // Constructor with dimensions
    template<typename... Dims>
    COOTensor(size_t initial_capacity, Dims... dims) 
        : capacity(initial_capacity), nnz_count(0) {
        static_assert(sizeof...(dims) == Order, "Number of dimensions must match Order");
        dimensions = {static_cast<size_t>(dims)...};
        
        // Allocate arrays for indices and values
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
        }
        values = new T[capacity];
    }

    // Constructor with dimensions array
    COOTensor(size_t initial_capacity, const std::array<size_t, Order>& dims) 
        : dimensions(dims), capacity(initial_capacity), nnz_count(0) {
        // Allocate arrays for indices and values
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
        }
        values = new T[capacity];
    }

    // Destructor
    ~COOTensor() {
        for (size_t i = 0; i < Order; ++i) {
            delete[] indices[i];
        }
        delete[] values;
    }

    // Copy constructor
    COOTensor(const COOTensor& other) 
        : dimensions(other.dimensions), capacity(other.capacity), nnz_count(other.nnz_count) {
        
        for (size_t i = 0; i < Order; ++i) {
            indices[i] = new size_t[capacity];
            std::memcpy(indices[i], other.indices[i], nnz_count * sizeof(size_t));
        }
        values = new T[capacity];
        std::memcpy(values, other.values, nnz_count * sizeof(T));
    }

    // Assignment operator
    COOTensor& operator=(const COOTensor& other) {
        if (this != &other) {
            // Free existing resources
            for (size_t i = 0; i < Order; ++i) {
                delete[] indices[i];
            }
            delete[] values;
            
            // Copy new data
            dimensions = other.dimensions;
            capacity = other.capacity;
            nnz_count = other.nnz_count;
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i] = new size_t[capacity];
                std::memcpy(indices[i], other.indices[i], nnz_count * sizeof(size_t));
            }
            values = new T[capacity];
            std::memcpy(values, other.values, nnz_count * sizeof(T));
        }
        return *this;
    }

    // Resize arrays when capacity is reached
    void resize(size_t new_capacity) {
        for (size_t dim = 0; dim < Order; ++dim) {
            size_t* new_indices = new size_t[new_capacity];
            std::memcpy(new_indices, indices[dim], nnz_count * sizeof(size_t));
            delete[] indices[dim];
            indices[dim] = new_indices;
        }
        
        T* new_values = new T[new_capacity];
        std::memcpy(new_values, values, nnz_count * sizeof(T));
        delete[] values;
        values = new_values;
        
        capacity = new_capacity;
    }

    // Add a non-zero element to the tensor
    template<typename... Idx>
    void add_element(T value, Idx... idx) {
        static_assert(sizeof...(idx) == Order, "Number of indices must match Order");
        
        std::array<size_t, Order> idx_array = {static_cast<size_t>(idx)...};
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        if (value != T(0)) {  // Only store non-zero values
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i][nnz_count] = idx_array[i];
            }
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    // Helper function to add element using array of indices
    void add_element_array(T value, const std::array<size_t, Order>& idx_array) {
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        if (value != T(0)) {
            if (nnz_count >= capacity) {
                resize(capacity * 2);
            }
            
            for (size_t i = 0; i < Order; ++i) {
                indices[i][nnz_count] = idx_array[i];
            }
            values[nnz_count] = value;
            nnz_count++;
        }
    }

    // Get the value at a specific position
    template<typename... Idx>
    T get(Idx... idx) const {
        static_assert(sizeof...(idx) == Order, "Number of indices must match Order");
        
        std::array<size_t, Order> idx_array = {static_cast<size_t>(idx)...};
        if (!check_bounds(idx_array, typename MakeIndexSequence<Order>::type{})) {
            throw std::out_of_range("Index out of bounds");
        }

        for (size_t i = 0; i < nnz_count; ++i) {
            bool match = true;
            for (size_t dim = 0; dim < Order; ++dim) {
                if (indices[dim][i] != idx_array[dim]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return values[i];
            }
        }
        return T(0);
    }

    // Get number of non-zero elements
    size_t nnz() const {
        return nnz_count;
    }

    size_t get_capacity() const {
        return capacity;
    }

    std::array<size_t, Order> get_dimensions() const {
        return dimensions;
    }

    std::array<size_t*, Order> get_indices() const {
        return indices;
    }

    T* get_values() const {
        return values;
    }    

    // Sort elements lexicographically by indices
    void sort() {
        size_t* idx = new size_t[nnz_count];
        for (size_t i = 0; i < nnz_count; ++i) {
            idx[i] = i;
        }

        std::sort(idx, idx + nnz_count,
            [this](size_t i1, size_t i2) {
                return this->compare_indices(i1, i2);
            });

        // Create temporary arrays for sorting
        std::array<size_t*, Order> new_indices;
        for (size_t dim = 0; dim < Order; ++dim) {
            new_indices[dim] = new size_t[capacity];
            for (size_t i = 0; i < nnz_count; ++i) {
                new_indices[dim][i] = indices[dim][idx[i]];
            }
        }

        T* new_values = new T[capacity];
        for (size_t i = 0; i < nnz_count; ++i) {
            new_values[i] = values[idx[i]];
        }

        // Swap pointers
        for (size_t dim = 0; dim < Order; ++dim) {
            delete[] indices[dim];
            indices[dim] = new_indices[dim];
        }
        delete[] values;
        values = new_values;
        delete[] idx;
    }

    // Print the tensor in COO format
    void print() const {
        std::cout << "COO Tensor (";
        for (size_t i = 0; i < Order; ++i) {
            std::cout << dimensions[i];
            if (i < Order - 1) std::cout << " x ";
        }
        std::cout << "), " << nnz_count << " non-zero elements:\n";

        for (size_t i = 0; i < nnz_count; ++i) {
            std::cout << "(";
            for (size_t dim = 0; dim < Order; ++dim) {
                std::cout << indices[dim][i];
                if (dim < Order - 1) std::cout << ", ";
            }
            std::cout << ") = " << values[i] << "\n";
        }
    }

    tblis::tensor<T> to_dense() const {
        size_t N = 1;
        for (size_t i = 0; i < Order; ++i) {
            N *= dimensions[i];
        }
        tblis::tensor<T> fullT(dimensions);
        std::fill(fullT.data(), fullT.data() + N, 0.0);
        size_t idt;
        size_t ddt;
        for (size_t i = 0; i < nnz_count; ++i) {
            idt = 0;
            for (size_t dim = 0; dim < Order; ++dim) {
                ddt = 1;
                for (size_t od = dim + 1; od < Order; ++od) {
                    ddt *= dimensions[od];
                }
                idt += indices[dim][i] * ddt;
            }
            fullT.data()[idt] = values[i];
        }
        return fullT;
    }

    // Simplified contraction function for single dimension
    template<size_t OtherOrder>
    COOTensor<T, Order + OtherOrder - 2> contract(
        const COOTensor<T, OtherOrder>& other,
        size_t this_dim,    // Dimension to contract from this tensor
        size_t other_dim    // Dimension to contract from other tensor
    ) const {
        if (this_dim >= Order || other_dim >= OtherOrder) {
            throw std::out_of_range("Contraction dimensions out of bounds");
        }
        
        // Calculate output dimensions
        constexpr size_t ResultOrder = Order + OtherOrder - 2;
        std::array<size_t, ResultOrder> out_dims;
        size_t out_idx = 0;
        
        // Map dimensions to output tensor
        for (size_t i = 0; i < Order; ++i) {
            if (i != this_dim) {
                out_dims[out_idx++] = dimensions[i];
            }
        }
        for (size_t i = 0; i < OtherOrder; ++i) {
            if (i != other_dim) {
                out_dims[out_idx++] = other.get_dimensions()[i];
            }
        }

        // Create result tensor
        size_t initial_capacity = std::min(nnz_count * other.nnz(), 
                                         capacity + other.get_capacity());
        COOTensor<T, ResultOrder> result(initial_capacity, out_dims);
        
        // For each non-zero element in this tensor
        for (size_t i = 0; i < nnz_count; ++i) {
            // For each non-zero element in other tensor
            for (size_t j = 0; j < other.nnz(); ++j) {
                // Check if contracted indices match
                if (indices[this_dim][i] == other.get_indices()[other_dim][j]) {
                    T prod = values[i] * other.get_values()[j];
                    if (prod != T(0)) {
                        // Build output indices
                        std::array<size_t, ResultOrder> out_indices;
                        out_idx = 0;
                        
                        // Map indices from this tensor
                        for (size_t k = 0; k < Order; ++k) {
                            if (k != this_dim) {
                                out_indices[out_idx++] = indices[k][i];
                            }
                        }
                        
                        // Map indices from other tensor
                        for (size_t k = 0; k < OtherOrder; ++k) {
                            if (k != other_dim) {
                                out_indices[out_idx++] = other.get_indices()[k][j];
                            }
                        }
                        
                        result.add_element_array(prod, out_indices);
                    }
                }
            }
        }
        
        return result;
    }
};

/*
// Helper class to represent a TT-core
template<typename T, size_t Order>
struct TTCore : public COOTensor<T, Order> {
    using COOTensor<T, Order>::COOTensor;  // Inherit constructors
};

// Function to contract a tensor train into a full tensor
template<typename T, typename... Cores>
auto contract_tt(const Cores&... cores) {
    static_assert(sizeof...(Cores) > 0, "Need at least one core");
    
    // Helper function to check if all arguments are TTCore
    constexpr bool all_are_tt_cores = (... && std::is_base_of_v<
        COOTensor<T, Cores::Order>, Cores>);
    static_assert(all_are_tt_cores, "All arguments must be TTCores");
    
    // Start with the first core
    auto result = std::get<0>(std::forward_as_tuple(cores...));
    
    // Sequentially contract with remaining cores
    size_t core_idx = 1;
    ((result = result.template contract(cores, 
        result.Order - 1,  // Last dimension of current result
        0)),              // First dimension of next core
        ...,
        core_idx++);
    
    return result;
}
*/

template <typename T>
auto SparseTTtoTensor3(COOTensor<T, 2> T1, COOTensor<T, 3> T2, COOTensor<T, 2> T3)
{
    auto i = T1.contract(T2, 1, 0);
    auto j = i.contract(T3, 2, 0);
    return j;
}

template <typename T>
auto SparseTTtoTensor4(COOTensor<T, 2> T1, COOTensor<T, 3> T2, COOTensor<T, 3> T3, COOTensor<T, 2> T4)
{
    auto i = T1.contract(T2, 1, 0);
    auto j = i.contract(T3, 2, 0);
    auto k = j.contract(T4, 3, 0);
    return k;
}

template <typename T>
auto SparseTTtoTensor5(COOTensor<T, 2> T1, COOTensor<T, 3> T2, COOTensor<T, 3> T3, COOTensor<T, 3> T4, COOTensor<T, 2> T5)
{
    auto i = T1.contract(T2, 1, 0);
    auto j = i.contract(T3, 2, 0);
    auto k = j.contract(T4, 3, 0);
    auto l = k.contract(T5, 4, 0);
    return l;
}