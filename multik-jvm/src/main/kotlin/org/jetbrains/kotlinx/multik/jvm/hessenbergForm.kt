package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.sqrt

internal data class MatrixPair(val a: NDArray<Double, D2>, val b: NDArray<Double, D2>)

internal data class VectorPair(val a: NDArray<Double, D1>, val b: NDArray<Double, D1>)



internal fun getReflector(x: VectorPair): Pair<Double, VectorPair> {
    val alpha: ComplexDouble = ComplexDouble(x.a[0], x.b[0])

    var xnorm = 0.0;
    for (i in 1 until x.a.size) {
        xnorm += x.a[i] * x.a[i] + x.b[i] * x.b[i]
    }
    xnorm = sqrt(xnorm)

    var v: VectorPair = VectorPair(mk.empty<Double, D1>(x.a.size), mk.empty<Double, D1>(x.a.size))
    v.a[0] = 1.0

    if(xnorm == 0.0) {
        return Pair(0.0, Object() as VectorPair)
    }
    TODO()
}

internal fun upperHessenberg(a: MatrixPair): Pair<MatrixPair, MatrixPair> {
    TODO();
}