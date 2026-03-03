cmake_minimum_required(VERSION 3.24)
project(multik_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

if (NOT DEFINED TARGET_OS)
    set(TARGET_OS ${PLATFORM})
endif ()

#### PARAMETERS FOR OPENBLAS ####
set(OPENBLAS_VERSION 0.3.31)

if (${TARGET_OS} STREQUAL "mingwX64" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(URL_OPENBLAS "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}-x64.zip")
else ()
    set(URL_OPENBLAS "https://github.com/OpenMathLib/OpenBLAS/archive/v${OPENBLAS_VERSION}.zip")
endif ()

set(CC ${CMAKE_C_COMPILER})
set(HOSTCC ${CC})
set(BINARY 64)
set(F_COMPILER GFORTRAN)
set(USE_OPENMP 0)
set(NUM_THREADS 64)
set(DYNAMIC_LIST )
set(NOFORTRAN 0)
set(TARGET_CPU )

if(${TARGET_OS} STREQUAL "androidArm64")
    set(suffix androidArm64)
    set(HOSTCC gcc)
    set(RANLIB llvm-ranlib)
    set(AR llvm-ar)
    set(NOFORTRAN 1)
    set(TARGET_CPU ARMV8)
elseif(${TARGET_OS} STREQUAL "linuxX64")
    set(suffix linuxX64)
    set(FC "gfortran -m64")
    set(FEXTRALIB -lgfortran)
    set(NO_AVX512 0)
    set(DYNAMIC_ARCH 1)
elseif (${TARGET_OS} STREQUAL "macosArm64" OR ${PLATFORM} MATCHES "Darwin-arm64")
    set(suffix macosArm64)
    set(FC "gfortran")
    set(FEXTRALIB "-lgfortran")
    set(NO_AVX512 1)
    set(DYNAMIC_ARCH 1)
elseif(${TARGET_OS} STREQUAL "macosX64" OR ${PLATFORM} MATCHES "Darwin-x86_64")
    set(suffix macosX64)
    set(FC "gfortran")
    set(FEXTRALIB "-lgfortran")
    set(NO_AVX512 0)
    set(DYNAMIC_ARCH 1)
# linuxArm64: uncomment when needed
# elseif(${TARGET_OS} STREQUAL "linuxArm64")
#     set(suffix linuxArm64)
#     set(FC "gfortran")
#     set(FEXTRALIB "-lgfortran")
#     set(NO_AVX512 1)
#     set(DYNAMIC_ARCH 1)
elseif (${TARGET_OS} MATCHES "mingwX64")
    set(suffix mingwX64)
    set(FC "gfortran -m64")
    set(FEXTRALIB "-lgfortran -lquadmath")
    set(NO_AVX512 1)
    set(DYNAMIC_ARCH 1)
else()
    message(WARNING "Platform ${PLATFORM} is not supported")
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
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
    #### INCLUDE OPENBLAS ####
    ExternalProject_Get_property(openblas SOURCE_DIR)
    set(OPENBLAS_INSTALL_DIR ${SOURCE_DIR})
    ########################
else ()
    find_program(MAKE_EXE NAMES make mingw32-make)
    if (NOT MAKE_EXE)
        message(FATAL_ERROR "Could not find 'make' executable (tried: make, mingw32-make). Please install it and ensure it is in your PATH.")
    endif ()

    set(OPENBLAS_BUILD_ARGS
            -C <SOURCE_DIR> -s
            CC=${CC}
            HOSTCC=${HOSTCC}
            BINARY=${BINARY}
            USE_OPENMP=${USE_OPENMP}
            NO_AVX512=${NO_AVX512}
            DYNAMIC_ARCH=${DYNAMIC_ARCH}
            DYNAMIC_LIST=${DYNAMIC_LIST}
            NUM_THREADS=${NUM_THREADS}
            BLAS3_MEM_ALLOC_THRESHOLD=32
    )
    if(TARGET_CPU)
        list(APPEND OPENBLAS_BUILD_ARGS TARGET=${TARGET_CPU})
    endif()
    if(RANLIB)
        list(APPEND OPENBLAS_BUILD_ARGS RANLIB=${RANLIB})
    endif()
    if(AR)
        list(APPEND OPENBLAS_BUILD_ARGS AR=${AR})
    endif()
    if(NOFORTRAN)
        list(APPEND OPENBLAS_BUILD_ARGS NOFORTRAN=1)
    else()
        list(APPEND OPENBLAS_BUILD_ARGS
                FC=${FC}
                F_COMPILER=${F_COMPILER}
                FEXTRALIB=${FEXTRALIB}
        )
    endif()

    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NOT NPROC OR NPROC EQUAL 0)
        set(NPROC 1)
    endif()

    ExternalProject_Add(openblas
            URL                 ${URL_OPENBLAS}
            PREFIX              ${CMAKE_CURRENT_BINARY_DIR}/openblas
            INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/openblas-install
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ${MAKE_EXE} -j${NPROC} ${OPENBLAS_BUILD_ARGS}
            INSTALL_COMMAND     ${MAKE_EXE} -C <SOURCE_DIR> install PREFIX=<INSTALL_DIR> CC=${CC}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
if(${TARGET_OS} STREQUAL "androidArm64")
    set(LIB_GFORTRAN "")
    set(LIB_QUADMATH "")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIB_GFORTRAN -static-libgfortran)
    set(LIB_QUADMATH "")
elseif(${TARGET_OS} STREQUAL "mingwX64")
    # Pre-built OpenBLAS is used on Windows; gfortran runtime is linked via -static in LIB_GCC
    set(LIB_GFORTRAN "")
    set(LIB_QUADMATH "")
else()
    find_program(GFORTRAN_EXECUTABLE NAMES gfortran)
    if (NOT GFORTRAN_EXECUTABLE)
        message(FATAL_ERROR "gfortran not found. It is required to locate libgfortran.a and libquadmath.a for linking.")
    endif ()

    execute_process(
        COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.a
        OUTPUT_VARIABLE LIB_GFORTRAN
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT LIB_GFORTRAN OR LIB_GFORTRAN STREQUAL "libgfortran.a")
        message(FATAL_ERROR "libgfortran.a not found by ${GFORTRAN_EXECUTABLE}. It is required to link multik_jni.")
    endif()

    execute_process(
        COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libquadmath.a
        OUTPUT_VARIABLE LIB_QUADMATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT LIB_QUADMATH OR LIB_QUADMATH STREQUAL "libquadmath.a")
        set(LIB_QUADMATH "")
    endif()
endif()

### TARGETS
if (${TARGET_OS} STREQUAL "mingwX64")
    set(LIB_GCC "-static-libgcc -static-libstdc++ -static -lpthread")
elseif(${TARGET_OS} STREQUAL "androidArm64")
    set(LIB_GCC "-ldl -lm")
else()
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
target_compile_features(${LIB_NAME} PRIVATE cxx_std_14)
target_compile_options(${LIB_NAME} PRIVATE
                                        -fPIC
                                        -O3
                                        -fno-exceptions
                                        -fno-math-errno
                                        -freciprocal-math
                                        -flto
                                        -funroll-loops
)
target_include_directories(${LIB_NAME} PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} optimized
        ${STATIC_OPENBLAS}
        ${LIB_GFORTRAN}
        ${LIB_QUADMATH}
        ${LIB_GCC}
)
add_dependencies(${LIB_NAME} openblas)
