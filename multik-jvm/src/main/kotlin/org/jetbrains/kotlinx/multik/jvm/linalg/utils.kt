package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

internal fun <T> requireSquare(a: MultiArray<T, D2>) {
    require(a.shape[0] == a.shape[1]) { "Square matrix expected, shape=(${a.shape[0]}, ${a.shape[1]}) given" }
}

