package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

internal fun <T : Any> requireSquare(a: MultiArray<T, D2>) {
    require(a.shape[0] == a.shape[1]) { "Square matrix expected, shape=(${a.shape[0]}, ${a.shape[1]}) given" }
}

// return hermitian transposed copy of matrix
internal fun NDArray<ComplexDouble, D2>.conjTranspose(): D2Array<ComplexDouble> {
    val ans = mk.empty<ComplexDouble, D2>(this.shape[1], this.shape[0])
    for (i in 0 until ans.shape[0]) {
        for (j in 0 until ans.shape[1]) {
            ans[i, j] = this[j, i].conjugate()
        }
    }
    return ans
}

fun Double.toComplexDouble(): ComplexDouble {
    return ComplexDouble(this, 0.0)
}

// TODO: remove
internal fun tempDot(a: NDArray<ComplexDouble, D2>, b: NDArray<ComplexDouble, D2>): D2Array<ComplexDouble> {
    require(a.shape[1] == b.shape[0]) { "Can't multiply" }

    val ans = mk.empty<ComplexDouble, D2>(a.shape[0], b.shape[1])

    for (i in 0 until a.shape[0]) {
        for (j in 0 until b.shape[1]) {
            for (k in 0 until a.shape[1]) {
                ans[i, j] += a[i, k] * b[k, j]
            }
        }
    }
    return ans
}

// computes some square root of complex number
// guarantee csqrt(a) * csqrt(a) = a
fun csqrt(a: ComplexDouble): ComplexDouble {
    val arg = a.angle()
    val absval = a.abs()
    return ComplexDouble(sqrt(absval) * cos(arg / 2), sqrt(absval) * sin(arg / 2))
}