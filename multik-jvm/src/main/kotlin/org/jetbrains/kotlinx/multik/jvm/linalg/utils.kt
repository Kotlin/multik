package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

internal fun requireSquare(shape: IntArray) {
    require(shape[0] == shape[1]) { "Square matrix expected, shape=(${shape[0]}, ${shape[1]}) given" }
}

// return hermitian transposed copy of matrix
@JvmName("conjTransposeFloat")
internal fun NDArray<ComplexFloat, D2>.conjTranspose(): D2Array<ComplexFloat> {
    val ans = mk.empty<ComplexFloat, D2>(this.shape[1], this.shape[0])
    for (i in 0 until ans.shape[0]) {
        for (j in 0 until ans.shape[1]) {
            ans[i, j] = this[j, i].conjugate()
        }
    }
    return ans
}

@JvmName("conjTransposeDouble")
internal fun NDArray<ComplexDouble, D2>.conjTranspose(): D2Array<ComplexDouble> {
    val ans = mk.empty<ComplexDouble, D2>(this.shape[1], this.shape[0])
    for (i in 0 until ans.shape[0]) {
        for (j in 0 until ans.shape[1]) {
            ans[i, j] = this[j, i].conjugate()
        }
    }
    return ans
}

internal fun requireDotShape(aShape: IntArray, bShape: IntArray) = require(aShape[1] == bShape[0]) {
    "Shapes mismatch: shapes " +
            "${aShape.joinToString(prefix = "(", postfix = ")")} and " +
            "${bShape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
            "${aShape[1]} (dim 1) != ${bShape[0]} (dim 0)"
}

// computes some square root of complex number
// guarantee csqrt(a) * csqrt(a) = a
fun csqrt(a: ComplexFloat): ComplexFloat {
    val arg = a.angle()
    val absval = a.abs()
    return ComplexFloat(sqrt(absval) * cos(arg / 2), sqrt(absval) * sin(arg / 2))
}

fun csqrt(a: ComplexDouble): ComplexDouble {
    val arg = a.angle()
    val absval = a.abs()
    return ComplexDouble(sqrt(absval) * cos(arg / 2), sqrt(absval) * sin(arg / 2))
}


/**
 * swap lines for plu decomposition
 * @param swap `a[i, j] = a[i + rowPerm[ i ], j].also { a[i + rowPerm[ i ], j] = a[i, j] }`
 */
internal inline fun swapLines(
    rowPerm: MultiArray<Int, D1>,
    from1: Int = 0, to1: Int = rowPerm.size, from2: Int = 0, to2: Int, swap: (Int, Int) -> Unit
) {
    for (i in from1 until to1) {
        if (rowPerm[i] != 0) {
            for (j in from2 until to2) {
                swap(i, j)
            }
        }
    }
}