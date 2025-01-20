cmake_minimum_required(VERSION 3.10)

# Project name
project(SparseTensorTrain CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA standard and architectures
#set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)  # Adjust based on the GPU

# Set default build type
set(default_build_type "Release")

# Include directory for headers
include_directories(include)

# Find libraries
find_package(BLAS REQUIRED)  # BLAS
find_package(LAPACK REQUIRED)  # LAPACK
find_library(LAPACKE_LIB lapacke REQUIRED)  # LAPACKE
find_library(TBLIS_LIB tblis REQUIRED)  # TBLIS (for fast dense tensor computation)
find_package(CUDA REQUIRED)  # Find CUDA packages

# Shared lib sources (dense api)
set(SHARED_SOURCES_UTIL
    src/util/linalgUtils.cpp
    src/util/timer.cpp)

set(SHARED_SOURCES_DENSETT
    src/algorithm/decomposition.cpp
    src/algorithm/interpolative.cpp
    src/algorithm/tensortrain.cpp
)

# Shared lib sources (sparse api)
set (SHARED_SOURCES_SPCUTT
    src/algorithm/spinterpolative.cpp
    src/algorithm/sptensortrain.cpp)

# Create a static library from shared sources
add_library(shared_lib_util STATIC ${SHARED_SOURCES_UTIL})
add_library(shared_lib_denseTT STATIC ${SHARED_SOURCES_DENSETT})
add_library(shared_lib_spcuTT STATIC ${SHARED_SOURCES_SPCUTT})

target_link_libraries(shared_lib_util ${BLAS_LIBRARIES} ${LAPACKE_LIB})
target_link_libraries(shared_lib_denseTT ${TBLIS_LIB} ${BLAS_LIBRARIES} ${LAPACKE_LIB})
target_link_libraries(shared_lib_spcuTT  ${TBLIS_LIB} ${BLAS_LIBRARIES} ${LAPACKE_LIB} 
                        ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
# Set CUDA compile options
target_compile_options(shared_lib_spcuTT PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Define sources
set(SOURCES_SYND src/synthetic_test/dense_test.cpp)
set(SOURCES_SYNS src/synthetic_test/sparse_test.cpp)
set(SOURCES_UT1 src/unit_test/unit_test_1.cpp)
set(SOURCES_UT2 src/unit_test/unit_test_2.cpp)
set(SOURCES_UT3 src/unit_test/unit_test_3.cpp)
set(SOURCES_UT4 src/unit_test/unit_test_4.cpp)
set(SOURCES_UT5 src/unit_test/unit_test_5.cpp)

# Executables for synthetic tests
add_executable(synthetic_dense_test ${SOURCES_SYND})
add_executable(synthetic_sparse_test ${SOURCES_SYNS})

# Link libraries for synthetic test exe
target_link_libraries(synthetic_dense_test PRIVATE shared_lib_denseTT shared_lib_util)
target_link_libraries(synthetic_sparse_test PRIVATE shared_lib_spcuTT shared_lib_denseTT shared_lib_util)

# Google Test setup
# If Google Test is not installed, download and build it
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# Prevent GoogleTest from building its own tests
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()  # Enable testing in CMake

# Executables for unit tests
add_executable(unit_test_1 ${SOURCES_UT1})
add_executable(unit_test_2 ${SOURCES_UT2})
add_executable(unit_test_3 ${SOURCES_UT3})
add_executable(unit_test_4 ${SOURCES_UT4})
add_executable(unit_test_5 ${SOURCES_UT5})

# Link libraries for unit test exe
target_link_libraries(unit_test_1 PRIVATE shared_lib_denseTT shared_lib_util gtest_main)
target_link_libraries(unit_test_2 PRIVATE shared_lib_denseTT shared_lib_util gtest_main)
target_link_libraries(unit_test_3 PRIVATE shared_lib_denseTT shared_lib_util gtest_main)
target_link_libraries(unit_test_4 PRIVATE shared_lib_denseTT shared_lib_util gtest_main)
target_link_libraries(unit_test_5 PRIVATE shared_lib_spcuTT  shared_lib_util gtest_main)

# Test cuda
add_executable(cutest src/util/cutest.cu)
set_target_properties(cutest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

# Output executables to different directories
set_target_properties(unit_test_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(cutest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(synthetic_dense_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)
set_target_properties(synthetic_sparse_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/synthetic_test)