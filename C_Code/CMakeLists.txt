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

# Define sources
set(SOURCES_1
    src/algo_test/synthetic_test.cpp
    src/algorithm/ttsvd.cpp
    src/algorithm/ttid.cpp)

set(SOURCES_2
    src/algo_test/frostt_test.cpp
    src/algorithm/ttsvd.cpp
    src/algorithm/ttid.cpp)

add_executable(synthetic_test ${SOURCES_1})
add_executable(frostt_test ${SOURCES_2})