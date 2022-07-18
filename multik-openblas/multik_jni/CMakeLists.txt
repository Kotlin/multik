cmake_minimum_required(VERSION 3.16)
project(multik_jni)

set(CMAKE_CXX_STANDARD 14)
set(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

if (NOT DEFINED TARGET_OS)
    set(TARGET_OS ${PLATFORM})
endif ()

#### PARAMETERS FOR OPENBLAS ####
set(OPENBLAS_VERSION 0.3.20)

if (${TARGET_OS} STREQUAL "mingwX64" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(URL_OPENBLAS "https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}-x64.zip")
else ()
    set(URL_OPENBLAS "https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.zip")
endif ()

set(CC ${CMAKE_C_COMPILER})
set(HOSTCC gcc)
set(BINARY 64)
set(F_COMPILER GFORTRAN)
set(CROSS_SUFFIX )
set(USE_OPENMP 0)
set(NUM_THREADS 64)

if (${TARGET_OS} MATCHES "androidArm64")
    set(suffix androidArm64)
    set(CC aarch64-linux-android-gcc)
    set(FC aarch64-linux-android-gfortran)
    set(CROSS_SUFFIX aarch64-linux-android-)
    set(NO_AVX512 1)
elseif (${TARGET_OS} MATCHES "linuxX64")
    set(suffix linuxX64)
    set(FC "gfortran -m64")
    set(FEXTRALIB -lgfortran)
    set(NO_AVX512 0)
    set(DYNAMIC_ARCH 1)
elseif (${TARGET_OS} MATCHES "macosArm64" OR ${PLATFORM} MATCHES "Darwin-arm64")
    set(suffix macosArm64)
    set(CC "/opt/homebrew/bin/gcc-11")
    set(FC "/opt/homebrew/bin/gfortran-11")
    set(FEXTRALIB "-lgfortran")
    set(NO_AVX512 1)
    set(DYNAMIC_ARCH 1)
elseif (${TARGET_OS} MATCHES "macosX64" OR ${PLATFORM} MATCHES "Darwin-x86_64")
    set(suffix macosX64)
    set(CC "/usr/local/bin/gcc-11")
    set(FC "/usr/local/bin/gfortran-11")
    set(FEXTRALIB "-lgfortran")
    set(NO_AVX512 1)
    set(DYNAMIC_ARCH 1)
elseif (${TARGET_OS} MATCHES "mingwX64")
    set(suffix mingwX64)
    set(FC "gfortran -m64")
    set(FEXTRALIB "-lgfortran -lquadmath")
    set(NO_AVX512 1)
    set(DYNAMIC_ARCH 1)
else()
    message(SEND_ERROR "Platform ${PLATFORM} is not supported")
    return()
endif ()
########################

#### DOWNLOAD AND INSTALL OPENBLAS ####
include(ExternalProject)
if (${TARGET_OS} STREQUAL "mingwX64")
    ExternalProject_Add(openblas
            URL                 ${URL_OPENBLAS}
            PREFIX              ${CMAKE_CURRENT_BINARY_DIR}/openblas
            SOURCE_DIR          ${CMAKE_CURRENT_BINARY_DIR}/openblas-install
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ""
            INSTALL_COMMAND     ""
            )
    #### INCLUDE OPENBLAS ####
    ExternalProject_Get_property(openblas SOURCE_DIR)
    set(OPENBLAS_INSTALL_DIR ${SOURCE_DIR})
    ########################
else ()
    find_program(MAKE_EXE NAMES make mingw32-make)
    ExternalProject_Add(openblas
            URL                 ${URL_OPENBLAS}
            PREFIX              ${CMAKE_CURRENT_BINARY_DIR}/openblas
            INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/openblas-install
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ${MAKE_EXE}
                                -C <SOURCE_DIR> -s
                                CROSS_SUFFIX=${CROSS_SUFFIX}
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
            INSTALL_COMMAND     ${MAKE_EXE} -C <SOURCE_DIR> install PREFIX=<INSTALL_DIR> CC=${CC}
            )
    #### INCLUDE OPENBLAS ####
    ExternalProject_Get_property(openblas INSTALL_DIR)
    set(OPENBLAS_INSTALL_DIR ${INSTALL_DIR})
endif()

include_directories(${OPENBLAS_INSTALL_DIR}/include)
set(STATIC_OPENBLAS ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a)
########################

#### VARS ####
include_directories("src/main/headers")
set(SRC_FILES_FOR_SHARED
        src/main/cpp/jni_Linalg.cpp
        src/main/cpp/jni_JniMath.cpp
        src/main/cpp/JniStat.cpp
        src/main/cpp/ComplexDouble.cpp
        src/main/cpp/ComplexFloat.cpp
        src/main/cpp/mk_math.cpp
        src/main/cpp/mk_linalg.cpp
        src/main/cpp/mk_stat.cpp
        )
set(SRC_FILES_FOR_STATIC
        src/main/cpp/mk_math.cpp
        src/main/cpp/mk_linalg.cpp
        src/main/cpp/mk_stat.cpp)
########################

### FIND GFORTRAN AND QUADMATH LIBRARIES ###
if (${GCC_LIB_PATH})
    message("Path to gcc libs does not defined")
endif ()
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIB_GFORTRAN -static-libgfortran)
else()
    find_library(LIB_GFORTRAN NAMES libgfortran.a PATHS ${GCC_LIB_PATH})
    if(NOT LIB_GFORTRAN)
        message("don't find libgfotran.a")
        set(LIB_GFORTRAN "${GCC_LIB_PATH}/libgfortran.a")
    endif ()
endif ()
if (${TARGET_OS} STREQUAL "mingwX64")
    set(LIB_QUADMATH "/msys64/mingw64/lib/libquadmath.a")
    set(LIB_GCC "-static-libgcc -static-libstdc++ -static -lpthread")
elseif(${TARGET_OS} STREQUAL "androidArm64")
    set(LIB_GCC "-static-libgcc -static-libstdc++ -Wl,--no-whole-archive -ldl -lm")
else()
    set(LIB_QUADMATH "${GCC_LIB_PATH}/libquadmath.a")
    set(LIB_GCC "-static-libgcc")
endif()

########################

#### Find JNI files ####
if (NOT ${TARGET_OS} STREQUAL androidArm64)
    find_package(JNI REQUIRED)
endif ()
########################

#### BUILD SHARED LIBRARY ####
set(LIB_NAME multik_jni-${suffix})

add_library(${LIB_NAME} SHARED ${SRC_FILES_FOR_SHARED})
target_compile_options(${LIB_NAME} PRIVATE -std=c++14
                                                    -fPIC
                                                    -O3
                                                    -fno-exceptions
                                                    -ffast-math)
target_include_directories(${LIB_NAME} PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} optimized
        ${STATIC_OPENBLAS}
        ${LIB_GFORTRAN}
        ${LIB_QUADMATH}
        ${LIB_GCC}
        )
add_dependencies(${LIB_NAME} openblas)
