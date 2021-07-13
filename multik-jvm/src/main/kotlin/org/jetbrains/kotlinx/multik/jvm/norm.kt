package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.ndarray.data.ImmutableMemoryView
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.pow

internal fun norm(
    mat: FloatArray, matOffset: Int, matStrides: IntArray,
    n: Int, m: Int, power: Int,
    consistent: Boolean
): Double {
    var result = 0.0

    val (matStride_0, matStride_1) = matStrides

    if (consistent) {
        for (element in mat) {
            result += (element.absoluteValue.toDouble()).pow(power)
        }
    } else {
        for (i in 0 until n) {
            val matInd = i * matStride_0 + matOffset
            for (k in 0 until m) {
                val elementDoubleAbsValue = mat[matInd + k * matStride_1].absoluteValue.toDouble()
                result += (elementDoubleAbsValue).pow(power)
            }
        }
    }

    return result.pow(1 / power.toDouble())
}

internal fun norm(
    mat: DoubleArray, matOffset: Int, matStrides: IntArray,
    n: Int, m: Int, power: Int,
    consistent: Boolean
): Double {
    //most common case of matrix elements
    var result = 0.0

    val (matStride_0, matStride_1) = matStrides

    if (consistent) {
        result = mat.sumOf { abs(it).pow(power) }
    } else {
        for (i in 0 until n) {
            val matInd = i * matStride_0 + matOffset
            for (k in 0 until m) {
                val elementDoubleAbsValue = abs(mat[matInd + k * matStride_1])
                result += (elementDoubleAbsValue).pow(power)
            }
        }
    }

    return result.pow(1 / power.toDouble())
}

internal fun <T : Number> norm(
    mat: ImmutableMemoryView<T>, matOffset: Int, matStrides: IntArray,
    n: Int, m: Int, power: Int,
    consistent: Boolean
): Double {
    var result = 0.0

    val (matStride_0, matStride_1) = matStrides

    if (consistent) {
        result = mat.sumOf { abs(it.toDouble()).pow(power) }
    } else {
        for (i in 0 until n) {
            val matInd = i * matStride_0 + matOffset
            for (k in 0 until m) {
                val elementDoubleAbsValue = abs(mat[matInd + k * matStride_1].toDouble())
                result += (elementDoubleAbsValue).pow(power)
            }
        }
    }

    return result.pow(1 / power.toDouble())
}