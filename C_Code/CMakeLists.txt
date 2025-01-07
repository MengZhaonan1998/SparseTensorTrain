cmake_minimum_required(VERSION 3.10)

# Project name
project(SparseTensorTrain CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA standard and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)  # Adjust based on your GPU

# Set default build type
set(default_build_type "Debug")

# Include directory for headers
include_directories(include)

# Find BLAS/LAPACK libraries
find_package(BLAS REQUIRED)     # BLAS
find_package(LAPACK REQUIRED)   # LAPACK
find_library(LAPACKE_LIB lapacke REQUIRED) # LAPACKE

#find_package(OpenMP REQUIRED)   # OPENMP

# Find CUDA packages
find_package(CUDAToolkit REQUIRED)

# TBLIS (for fast dense tensor computation)
find_library(TBLIS_LIB tblis REQUIRED)     

# ==== SUPERLU OPENMP MT (for fast sparse lu decomposition) ==== #
#find_library(SUPERLU_LIBRARY 
#    NAMES superlu_mt_OPENMP
#    PATHS 
#        /usr/local/lib
#        /usr/lib
#) 
#find_path(SUPERLU_INCLUDE_DIR
#    NAMES superlu/slu_ddefs.h
#    PATHS
#        /usr/local/include
#        /usr/include
#)
# Print out all variables to help diagnose
#message(STATUS "SUPERLU_LIBRARY: ${SUPERLU_LIBRARY}")
#message(STATUS "SUPERLU_INCLUDE_DIR: ${SUPERLU_INCLUDE_DIR}")
#message(STATUS "SuperLU_FOUND: ${SuperLU_FOUND}")
# Manually check library existence
#if(EXISTS "${SUPERLU_LIBRARY}")
#    message(STATUS "Library file exists: ${SUPERLU_LIBRARY}")
#else()
#    message(FATAL_ERROR "Library file does NOT exist: ${SUPERLU_LIBRARY}")
#endif()
# ============================================================== #

# Shared sources
set(SHARED_SOURCES_DENSETT
    src/algorithm/linalgUtils.cpp
    src/algorithm/decomposition.cpp
    src/algorithm/interpolative.cpp
    src/algorithm/tensortrain.cpp
)

set (SHARED_SOURCES_SPCUTT
    src/algorithm/spinterpolative.cpp
    src/algorithm/sparseops.cu
    src/algorithm/sptensortrain.cpp)

# Create a static library from shared sources
add_library(shared_lib_denseTT STATIC ${SHARED_SOURCES_DENSETT})
add_library(shared_lib_spcuTT STATIC ${SHARED_SOURCES_SPCUTT})

# Define sources
set(SOURCES_SYND src/synthetic_test/dense_test.cpp)
set(SOURCES_SYNS src/synthetic_test/sparse_test.cpp)
set(SOURCES_UT1 src/unit_test/unit_test_1.cpp)
set(SOURCES_UT2 src/unit_test/unit_test_2.cpp)
set(SOURCES_UT3 src/unit_test/unit_test_3.cpp)
set(SOURCES_UT4 src/unit_test/unit_test_4.cpp)

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

add_executable(cuda_hello src/algorithm/hello.cu)
set_target_properties(cuda_hello PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

# Executables for synthetic tests
add_executable(synthetic_dense_test ${SOURCES_SYND})
add_executable(synthetic_sparse_test ${SOURCES_SYNS})

# Executables for unit tests
add_executable(unit_test_1 ${SOURCES_UT1})
add_executable(unit_test_2 ${SOURCES_UT2})
add_executable(unit_test_3 ${SOURCES_UT3})
add_executable(unit_test_4 ${SOURCES_UT4})

# Link the shared library and other dependencies to executables
target_link_libraries(shared_lib_denseTT PRIVATE LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})
target_link_libraries(shared_lib_spcuTT PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cusparse PRIVATE LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})
# Set CUDA compile options
target_compile_options(shared_lib_spcuTT PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(synthetic_dense_test PRIVATE shared_lib_denseTT)
target_link_libraries(synthetic_sparse_test PRIVATE shared_lib_spcuTT)
target_link_libraries(unit_test_1 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_2 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_3 PRIVATE shared_lib_denseTT gtest_main)
target_link_libraries(unit_test_4 PRIVATE shared_lib_denseTT gtest_main)

# Output unit-test executable to unit_test/
set_target_properties(unit_test_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)

