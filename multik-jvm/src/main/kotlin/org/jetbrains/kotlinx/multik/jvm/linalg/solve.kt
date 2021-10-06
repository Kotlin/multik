/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.abs

internal fun solveDouble(
    a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, singularityErrorLevel: Double = 1e-7
): D2Array<Double> {
    val (P, L, U) = pluCompressed(a)
    val _b = b.deepCopy() as D2Array<Double>

    swapLines(P, to2 = _b.shape[1]) { i, k ->
        _b[i, k] = _b[i + P[i], k].also { _b[i + P[i], k] = _b[i, k] }
    }
    for (i in 0 until U.shape[0]) {
        if (abs(U[i, i]) < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangle(U, solveTriangle(L, _b), false)
}

internal fun solveFloat(
    a: MultiArray<Float, D2>, b: MultiArray<Float, D2>, singularityErrorLevel: Float = 1e-6f
): D2Array<Float> {
    val (P, L, U) = pluCompressed(a)
    val _b = b.deepCopy() as D2Array<Float>

    swapLines(P, to2 = _b.shape[1]) { i, k ->
        _b[i, k] = _b[i + P[i], k].also { _b[i + P[i], k] = _b[i, k] }
    }
    for (i in 0 until U.shape[0]) {
        if (abs(U[i, i]) < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleF(U, solveTriangleF(L, _b), false)
}

internal fun solveComplexDouble(
    a: MultiArray<ComplexDouble, D2>, b: MultiArray<ComplexDouble, D2>, singularityErrorLevel: Double = 1e-7
): D2Array<ComplexDouble> {
    val (P, L, U) = pluCompressed(a)
    val _b = b.deepCopy() as D2Array<ComplexDouble>

    swapLines(P, to2 = _b.shape[1]) { i, k ->
        _b[i, k] = _b[i + P[i], k].also { _b[i + P[i], k] = _b[i, k] }
    }
    for (i in 0 until U.shape[0]) {
        if (U[i, i].abs() < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleComplexDouble(U, solveTriangleComplexDouble(L, _b), false)
}

internal fun solveComplexFloat(
    a: MultiArray<ComplexFloat, D2>, b: MultiArray<ComplexFloat, D2>, singularityErrorLevel: Float = 1e-6f
): D2Array<ComplexFloat> {
    val (P, L, U) = pluCompressed(a)
    val _b = b.deepCopy() as D2Array<ComplexFloat>

    swapLines(P, to2 = _b.shape[1]) { i, k ->
        _b[i, k] = _b[i + P[i], k].also { _b[i + P[i], k] = _b[i, k] }
    }
    for (i in 0 until U.shape[0]) {
        if (U[i, i].abs() < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleComplexFloat(U, solveTriangleComplexFloat(L, _b), false)
}


private fun solveTriangle(
    a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, isLowerTriangle: Boolean = true
): D2Array<Double> = solveTriangleleCommon(a, b.deepCopy() as D2Array<Double>, isLowerTriangle,
    { cf1, cf2 -> cf1 / cf2 }, { cf1, cf2, cf3, cf4 -> cf1 - (cf2 * cf3 / cf4) })

private fun solveTriangleF(
    a: MultiArray<Float, D2>, b: MultiArray<Float, D2>, isLowerTriangle: Boolean = true
): D2Array<Float> = solveTriangleleCommon(a, b.deepCopy() as D2Array<Float>, isLowerTriangle,
    { cf1, cf2 -> cf1 / cf2 }, { cf1, cf2, cf3, cf4 -> cf1 - (cf2 * cf3 / cf4) })

private fun solveTriangleComplexDouble(
    a: MultiArray<ComplexDouble, D2>, b: MultiArray<ComplexDouble, D2>, isLowerTriangle: Boolean = true
): D2Array<ComplexDouble> = solveTriangleleCommon(a, b.deepCopy() as D2Array<ComplexDouble>, isLowerTriangle,
    { cf1, cf2 -> cf1 / cf2 }, { cf1, cf2, cf3, cf4 -> cf1 - (cf2 * cf3 / cf4) })

private fun solveTriangleComplexFloat(
    a: MultiArray<ComplexFloat, D2>, b: MultiArray<ComplexFloat, D2>, isLowerTriangle: Boolean = true
): D2Array<ComplexFloat> = solveTriangleleCommon(a, b.deepCopy() as D2Array<ComplexFloat>, isLowerTriangle,
    { cf1, cf2 -> cf1 / cf2 }, { cf1, cf2, cf3, cf4 -> cf1 - (cf2 * cf3 / cf4) })

/**
 * solves a*x = b where a lower or upper triangle square matrix
 * @param actionFirst x[i, j] /= a[i, i]
 * @param actionSecond x[k, j] -= a[k, i] * x[i, j] / a[k, k]
 */
private inline fun <T> solveTriangleleCommon(
    a: MultiArray<T, D2>, x: D2Array<T>,
    isLowerTriangle: Boolean = true, actionFirst: (T, T) -> T, actionSecond: (T, T, T, T) -> T
): D2Array<T> {
    for (i in 0 until x.shape[0]) {
        for (j in 0 until x.shape[1]) {
            x[i, j] = actionFirst(x[i, j], a[i, i])
        }
    }

    if (isLowerTriangle) {
        for (i in 0 until x.shape[0]) {
            for (k in i + 1 until x.shape[0]) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] = actionSecond(x[k, j], a[k, i], x[i, j], a[k, k])
                }
            }
        }
    } else {
        for (i in x.shape[0] - 1 downTo 0) {
            for (k in i - 1 downTo 0) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] = actionSecond(x[k, j], a[k, i], x[i, j], a[k, k])
                }
            }
        }
    }
    return x
}