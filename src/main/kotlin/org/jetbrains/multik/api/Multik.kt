package org.jetbrains.multik.api

/**
 * Alternative names.
 */
typealias mk = Multik
typealias mk_math = Multik.Math
typealias mk_linalg = Multik.LinAlg

/**
 * The basic object through which calls all ndarray functions.
 *
 * TODO(Add load native library)
 */
object Multik {

    init {
        System.load("/Users/pavel.gorgulov/Projects/main_project/multik/src/main/cpp/cmake-build-debug/libcpp.dylib")
    }

    object Math
    object LinAlg
}
