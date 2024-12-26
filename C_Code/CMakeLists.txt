cmake_minimum_required(VERSION 3.10)

# Project name
project(SparseTensorTrain)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type
set(default_build_type "Debug")

# Include directory for headers
include_directories(include)
#include_directories(/home/mengzn/Desktop/tblis/src)

# Find libraries
find_package(BLAS REQUIRED)     # BLAS
find_package(LAPACK REQUIRED)   # LAPACK
find_package(OpenMP REQUIRED)   # OPENMP
find_library(LAPACKE_LIB lapacke REQUIRED) # LAPACKE

# TBLIS (for fast dense tensor computation)
find_library(TBLIS_LIB tblis REQUIRED)     
# ...tblis...

# ==== SUPERLU OPENMP MT (for fast sparse lu decomposition) ==== #
find_library(SUPERLU_LIBRARY 
    NAMES superlu_mt_OPENMP
    PATHS 
        /usr/local/lib
        /usr/lib
) 
find_path(SUPERLU_INCLUDE_DIR
    NAMES superlu/slu_ddefs.h
    PATHS
        /usr/local/include
        /usr/include
)
# Print out all variables to help diagnose
message(STATUS "SUPERLU_LIBRARY: ${SUPERLU_LIBRARY}")
message(STATUS "SUPERLU_INCLUDE_DIR: ${SUPERLU_INCLUDE_DIR}")
message(STATUS "SuperLU_FOUND: ${SuperLU_FOUND}")
# Manually check library existence
if(EXISTS "${SUPERLU_LIBRARY}")
    message(STATUS "Library file exists: ${SUPERLU_LIBRARY}")
else()
    message(FATAL_ERROR "Library file does NOT exist: ${SUPERLU_LIBRARY}")
endif()
# ============================================================== #

# Define sources
set(SOURCES_1
    src/synthetic_test/dense_test.cpp
    src/algorithm/tensortrain.cpp
    src/algorithm/linalgUtils.cpp
    src/algorithm/decomposition.cpp
    src/algorithm/interpolative.cpp)

#set(SOURCES_2
#    src/algo_test/frostt_test.cpp
#    src/algorithm/tensortrain.cpp
#    src/algorithm/linalgUtils.cpp
#    src/algorithm/decomposition.cpp
#    src/algorithm/interpolative.cpp)

set(SOURCES_3
    src/unit_test/unit_test_1.cpp
    src/algorithm/linalgUtils.cpp
    src/algorithm/decomposition.cpp
    src/algorithm/interpolative.cpp)

set(SOURCES_4
    src/unit_test/unit_test_2.cpp
    src/algorithm/linalgUtils.cpp)

set(SOURCES_5
    src/unit_test/unit_test_3.cpp
    src/algorithm/tensortrain.cpp
    src/algorithm/linalgUtils.cpp
    src/algorithm/decomposition.cpp
    src/algorithm/interpolative.cpp)

add_executable(synthetic_dense_test ${SOURCES_1})
#add_executable(frostt_test ${SOURCES_2})

target_link_libraries(synthetic_dense_test PRIVATE LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})
#target_link_libraries(frostt_test PRIVATE LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})

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

# ==== UNIT TEST ==== #
# Define the test executable
add_executable(unit_test_1 ${SOURCES_3})
add_executable(unit_test_2 ${SOURCES_4})
add_executable(unit_test_3 ${SOURCES_5})
#add_executable(unit_test_4 ${SOURCES_6})

# Link Google Test libraries and other dependencies (LAPACK/BLAS)
target_link_libraries(unit_test_1 PRIVATE gtest_main ${LAPACKE_LIB} BLAS::BLAS )
target_link_libraries(unit_test_2 PRIVATE gtest_main ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})
target_link_libraries(unit_test_3 PRIVATE gtest_main ${LAPACKE_LIB} BLAS::BLAS ${TBLIS_LIB})
#target_link_libraries(unit_test_4 PRIVATE gtest_main ${LAPACKE_LIB} BLAS::BLAS ${SUPERLU_LIBRARY} OpenMP::OpenMP_CXX)

# Output unit-test executable to unit_test/
set_target_properties(unit_test_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
set_target_properties(unit_test_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)

