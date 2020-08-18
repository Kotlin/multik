cmake_minimum_required(VERSION 3.16)
project(cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(JNI REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${JNI_INCLUDE_DIRS})

#find_package(LAPACK REQUIRED)
#find_package(BLAS REQUIRED)
#link_libraries(/usr/local/Cellar/lapack/3.9.0_1/lib/liblapacke.dylib)
#link_libraries(/usr/local/Cellar/openblas/0.3.10_1/lib/libblas.dylib)

include_directories(../../build/openblas/include/)
link_libraries(../../build/openblas/lib/libopenblas.dylib)

#find_package(nsimd_x86_64 REQUIRED)
#set(Boost_SIMD_INCLUDE_DIR /Users/pavel.gorgulov/Projects/simd/nsimd/include)
#include_directories(${Boost_SIMD_INCLUDE_DIR})
#find_package(Boost.SIMD REQUIRED)
#include_directories(${"Boost.SIMD@_INCLUDE_DIRS"})

include_directories("headers/")
set(SRC_FILES
        cpp/jni_Linalg.cpp
        cpp/org_jetbrains_multik_jni_JniMath.cpp
        )

add_library(jni_multik SHARED ${SRC_FILES})