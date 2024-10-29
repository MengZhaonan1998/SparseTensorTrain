cmake_minimum_required(VERSION 3.8)

# Project name
project(SparseTensorTrain)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type
set(default_build_type "Debug")

# Include directory for headers
include_directories(include)

# Find libraries
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_library(LAPACKE_LIB lapacke REQUIRED)

# Define sources
set(SOURCES_1
    src/algo_test/synthetic_test.cpp
    src/algorithm/ttsvd.cpp
    src/algorithm/ttid.cpp)

set(SOURCES_2
    src/algo_test/frostt_test.cpp
    src/algorithm/ttsvd.cpp
    src/algorithm/ttid.cpp)

set(SOURCES_3
    src/unit_test/unit_test_1.cpp
    src/algorithm/ttsvd.cpp
    src/algorithm/ttid.cpp)

add_executable(synthetic_test ${SOURCES_1})
add_executable(frostt_test ${SOURCES_2})

target_link_libraries(synthetic_test PRIVATE LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS)


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

# Define the test executable
add_executable(unit_test_1 ${SOURCES_3})

# Link Google Test libraries and other dependencies (LAPACK/BLAS)
target_link_libraries(unit_test_1 PRIVATE gtest_main LAPACK::LAPACK ${LAPACKE_LIB} BLAS::BLAS)

