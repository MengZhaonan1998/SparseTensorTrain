cmake_minimum_required(VERSION 3.8)

set(default_build_type "Debug")

project(ttsvd LANGUAGES CXX)

add_executable(ttsvd src/ttsvd.cpp)

target_compile_features(ttsvd PUBLIC cxx_std_14)