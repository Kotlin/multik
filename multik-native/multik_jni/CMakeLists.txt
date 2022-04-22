cmake_minimum_required(VERSION 3.16)
project(multik_jni)

set(CMAKE_CXX_STANDARD 14)

set(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

#### PARAMETERS FOR OPENBLAS ####
set(OPENBLAS_VERSION 0.3.20)

if (${PLATFORM} MATCHES "Windows-x86_64")
    set(URL_OPENBLAS "https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}-x64.zip")
else ()
    set(URL_OPENBLAS "https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.zip")
endif ()

set(CC ${CMAKE_C_COMPILER})
set(HOSTCC gcc)
set(BINARY 64)
set(F_COMPILER GFORTRAN)
set(USE_OPENMP 0)
set(DYNAMIC_ARCH 1)
set(NUM_THREADS 64)

if (${PLATFORM} MATCHES "android-arm64")
elseif (${PLATFORM} MATCHES "ios-arm")
elseif (${PLATFORM} MATCHES "ios-arm64")
elseif (${PLATFORM} MATCHES "Linux-x86_64")
    set(FC "gfortran -m64")
    set(FEXTRALIB -lgfortran)
    set(NO_AVX512 0)
elseif (${PLATFORM} MATCHES "Darwin-arm64")
elseif (${PLATFORM} MATCHES "Darwin-x86_64")
    set(CC "/usr/local/bin/gcc-11")
    set(FC "/usr/local/bin/gfortran-11")
    set(FEXTRALIB "-lgfortran")
    set(NO_AVX512 1)
elseif (${PLATFORM} MATCHES "Windows-x86_64")
    set(FC "gfortran -m64")
    set(FEXTRALIB "-lgfortran -lquadmath")
    set(NO_AVX512 1)
else()
    message(SEND_ERROR "Platform ${PLATFORM} is not supported")
    return()
endif ()
########################

#### DOWNLOAD AND INSTALL OPENBLAS ####
include(ExternalProject)
find_program(MAKE_EXE NAMES make)
ExternalProject_Add(openblas
        URL                 ${URL_OPENBLAS}
        PREFIX              ${CMAKE_CURRENT_BINARY_DIR}/openblas
        INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/openblas-install
        CONFIGURE_COMMAND   ""
        BUILD_COMMAND       make
                            -C <SOURCE_DIR> -s -j
                            CC=${CC}
                            FC=${FC}
                            HOSTCC=${HOSTCC}
                            BINARY=${BINARY}
                            F_COMPILER=${F_COMPILER}
                            FEXTRALIB=${FEXTRALIB}
                            USE_OPENMP=${USE_OPENMP}
                            NO_AVX512=${NO_AVX512}
                            DYNAMIC_ARCH=${DYNAMIC_ARCH}
                            NUM_THREADS=${NUM_THREADS}
        INSTALL_COMMAND     make -C <SOURCE_DIR> install PREFIX=<INSTALL_DIR> CC=${CC}
        )
########################

#### INCLUDE OPENBLAS ####
ExternalProject_Get_property(openblas INSTALL_DIR)
set(OPENBLAS_INSTALL_DIR ${INSTALL_DIR})
include_directories(${OPENBLAS_INSTALL_DIR}/include)
set(STATIC_OPENBLAS ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a)
########################

#### VARS ####
include_directories("src/main/headers")
set(SRC_FILES_FOR_SHARED
        src/main/cpp/jni_Linalg.cpp
        src/main/cpp/jni_JniMath.cpp
        src/main/cpp/ComplexDouble.cpp
        src/main/cpp/ComplexFloat.cpp
        src/main/cpp/mk_math.cpp
        src/main/cpp/mk_linalg.cpp
        )
set(SRC_FILES_FOR_STATIC
        src/main/cpp/mk_math.cpp
        src/main/cpp/mk_linalg.cpp)
########################

#### BUILD INTERMEDIATE STATIC LIBRARY ####
add_library(static_multik_jni STATIC ${SRC_FILES_FOR_STATIC})
target_compile_options(static_multik_jni PRIVATE -std=c++14
                                                -O3
                                                -fno-exceptions
                                                -ffast-math)
add_dependencies(static_multik_jni openblas)
########################

### FIND GFORTRAN AND QUADMATH LIBRARIES ###
if (${GCC_LIB_PATH})
    message("Path to gcc libs does not defined")
endif ()
find_library(LIB_GFORTRAN NAMES libgfortran.a PATHS ${GCC_LIB_PATH})
if(NOT LIB_GFORTRAN)
    message("don't find libgfotran.a")
    set(LIB_GFORTRAN "${GCC_LIB_PATH}/libgfortran.a")
endif ()
set(LIB_QUADMATH "${GCC_LIB_PATH}/libquadmath.a")
########################

#### BUILD FAT STATIC LIBRARY ####
add_custom_target(bundled_static_multik_jni
        COMMAND libtool -static -o libmultik_jni.a $<TARGET_FILE:static_multik_jni> ${STATIC_OPENBLAS} ${LIB_GFORTRAN} ${LIB_QUADMATH} ${LIB_GCC}
        )
########################

#### Find JNI files ####
find_package(JNI REQUIRED)
if (NOT JNI_FOUND)
    message(SEND_ERROR "Failed to find JNI")
    return()
endif ()
########################

#### BUILD SHARED LIBRARY ####
add_library(multik_jni SHARED ${SRC_FILES_FOR_SHARED})
target_compile_options(multik_jni PRIVATE -std=c++14
                                                    -fPIC
                                                    -O3
                                                    -fno-exceptions
                                                    -ffast-math)
target_include_directories(multik_jni PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries(multik_jni ${STATIC_OPENBLAS})
target_link_libraries(multik_jni ${LIB_GFORTRAN})
target_link_libraries(multik_jni ${LIB_QUADMATH})
target_link_libraries(multik_jni -static-libgcc)
add_dependencies(multik_jni openblas)
