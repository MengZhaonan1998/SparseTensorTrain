// core.h - Core definitions and basic utilities
#ifndef TENSOR_CORE_H
#define TENSOR_CORE_H

// External
#include <tblis/tblis.h>
#include <cblas.h>
#include <lapacke.h>
#include <cuda_runtime.h>

// STD
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <array>
#include <iomanip>
#include <fstream>
#include <typeinfo>
#include <omp.h>

#define assertm(exp, msg) assert(((void)msg, exp)) // Use (void) to silence unused warnings.

#endif // TENSOR_CORE_H