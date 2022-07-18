/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.kotlin.linalg

import kotlin.math.abs
import kotlin.math.pow

@Suppress("DuplicatedCode")
internal fun norm(
    mat: FloatArray, matOffset: Int, matStrides: IntArray, n: Int, m: Int, power: Int, consistent: Boolean
): Float {
    var result = 0.0

    val (matStride_0, matStride_1) = matStrides

    if (consistent) {
        for (element in mat) {
            result += abs(element).pow(power)
        }
    } else {
        for (i in 0 until n) {
            val matInd = i * matStride_0 + matOffset
            for (k in 0 until m) {
                val absValue = abs(mat[matInd + k * matStride_1])
                result += absValue.pow(power)
            }
        }
    }

    return result.pow(1.0 / power).toFloat()
}

@Suppress("DuplicatedCode")
internal fun norm(
    mat: DoubleArray, matOffset: Int, matStrides: IntArray, n: Int, m: Int, power: Int, consistent: Boolean
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
                val absValue = abs(mat[matInd + k * matStride_1])
                result += absValue.pow(power)
            }
        }
    }

    return result.pow(1.0 / power)
}

//internal fun <T : Number> norm(
//    mat: ImmutableMemoryView<T>, matOffset: Int, matStrides: IntArray, n: Int, m: Int, power: Int, consistent: Boolean
//): Double {
//    var result = 0.0
//
//    val (matStride_0, matStride_1) = matStrides
//
//    if (consistent) {
//        result = mat.sumOf { abs(it.toDouble()).pow(power) }
//    } else {
//        for (i in 0 until n) {
//            val matInd = i * matStride_0 + matOffset
//            for (k in 0 until m) {
//                val absValue = abs(mat[matInd + k * matStride_1].toDouble())
//                result += absValue.pow(power)
//            }
//        }
//    }
//
//    return result.pow(1.0 / power)
//}