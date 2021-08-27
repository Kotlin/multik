package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import kotlin.math.abs


//---------Double and Float case----------------

internal fun solveDouble(a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, singularityErrorLevel: Double = 1e-7): D2Array<Double> {
    requireSquare(a)
    require(a.shape[1] == b.shape[0])
    { "Shapes of arguments are incompatible: expected a.shape[1] = ${a.shape[1]} to be equal to the b.shape[0] = ${b.shape[0]}" }
    val (P, L, U) = pluCompressed(a)
    val _b: D2Array<Double> = (b as D2Array<Double>).deepCopy()

    for (i in P.indices) {
        if (P[i] != 0) {
            _b[i] = _b[i + P[i]].deepCopy().also { _b[i + P[i]] = _b[i].deepCopy() }
        }
    }
    for (i in 0 until U.shape[0]) {
        if (abs(U[i, i]) < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangle(U, solveTriangle(L, _b), false)
}

internal fun solveFloat(a: MultiArray<Float, D2>, b: MultiArray<Float, D2>, singularityErrorLevel: Float = 1e-6f): D2Array<Float> {
    requireSquare(a)
    require(a.shape[1] == b.shape[0])
    { "Shapes of arguments are incompatible: expected a.shape[1] = ${a.shape[1]} to be equal to the b.shape[0] = ${b.shape[0]}" }
    val (P, L, U) = pluCompressedF(a)
    val _b: D2Array<Float> = (b as D2Array<Float>).deepCopy()

    for (i in P.indices) {
        if (P[i] != 0) {
            _b[i] = _b[i + P[i]].deepCopy().also { _b[i + P[i]] = _b[i].deepCopy() }
        }
    }
    for (i in 0 until U.shape[0]) {
        if (abs(U[i, i]) < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleF(U, solveTriangleF(L, _b), false)
}

/**
 * solves a*x = b where a lower or upper triangle square matrix
 */
private fun solveTriangle(a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, isLowerTriangle: Boolean = true): D2Array<Double> {
    require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
    requireSquare(a)

    val x = b.map { it }
    for (i in 0 until x.shape[0]) {
        for (j in 0 until x.shape[1]) {
            x[i, j] /= a[i, i]
        }
    }
    if (isLowerTriangle) {
        for (i in 0 until x.shape[0]) {
            for (k in i + 1 until x.shape[0]) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    } else {
        for (i in x.shape[0] - 1 downTo 0) {
            for (k in i - 1 downTo 0) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    }
    return x
}

private fun solveTriangleF(a: MultiArray<Float, D2>, b: MultiArray<Float, D2>, isLowerTriangle: Boolean = true): D2Array<Float> {
    require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
    requireSquare(a)

    val x = b.map { it }
    for (i in 0 until x.shape[0]) {
        for (j in 0 until x.shape[1]) {
            x[i, j] /= a[i, i]
        }
    }
    if (isLowerTriangle) {
        for (i in 0 until x.shape[0]) {
            for (k in i + 1 until x.shape[0]) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    } else {
        for (i in x.shape[0] - 1 downTo 0) {
            for (k in i - 1 downTo 0) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    }
    return x
}


//---------Complex case----------------

internal fun solveComplexDouble(a: MultiArray<ComplexDouble, D2>, b: MultiArray<ComplexDouble, D2>, singularityErrorLevel: Double = 1e-7): D2Array<ComplexDouble> {
    requireSquare(a)
    require(a.shape[1] == b.shape[0])
    { "Shapes of arguments are incompatible: expected a.shape[1] = ${a.shape[1]} to be equal to the b.shape[0] = ${b.shape[0]}" }
    val (P, L, U) = pluCompressedComplexDouble(a)
    val _b: D2Array<ComplexDouble> = (b as D2Array<ComplexDouble>).deepCopy()

    for (i in P.indices) {
        if (P[i] != 0) {
            _b[i] = _b[i + P[i]].deepCopy().also { _b[i + P[i]] = _b[i].deepCopy() }
        }
    }
    for (i in 0 until U.shape[0]) {
        if (U[i, i].abs() < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleComplexDouble(U, solveTriangleComplexDouble(L, _b), false)
}

internal fun solveComplexFloat(a: MultiArray<ComplexFloat, D2>, b: MultiArray<ComplexFloat, D2>, singularityErrorLevel: Float = 1e-6f): D2Array<ComplexFloat> {
    requireSquare(a)
    require(a.shape[1] == b.shape[0])
    { "Shapes of arguments are incompatible: expected a.shape[1] = ${a.shape[1]} to be equal to the b.shape[0] = ${b.shape[0]}" }
    val (P, L, U) = pluCompressedComplexFloat(a)
    val _b: D2Array<ComplexFloat> = (b as D2Array<ComplexFloat>).deepCopy()

    for (i in P.indices) {
        if (P[i] != 0) {
            _b[i] = _b[i + P[i]].deepCopy().also { _b[i + P[i]] = _b[i].deepCopy() }
        }
    }
    for (i in 0 until U.shape[0]) {
        if (U[i, i].abs() < singularityErrorLevel) {
            throw ArithmeticException("Matrix a is singular or almost singular")
        }
    }

    return solveTriangleComplexFloat(U, solveTriangleComplexFloat(L, _b), false)
}

/**
 * solves a*x = b where a lower or upper triangle square matrix
 */
private fun solveTriangleComplexDouble(a: MultiArray<ComplexDouble, D2>, b: MultiArray<ComplexDouble, D2>, isLowerTriangle: Boolean = true): D2Array<ComplexDouble> {
    require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
    requireSquare(a)

    val x = b.deepCopy() as D2Array<ComplexDouble>
    for (i in 0 until x.shape[0]) {
        for (j in 0 until x.shape[1]) {
            x[i, j] /= a[i, i]
        }
    }
    if (isLowerTriangle) {
        for (i in 0 until x.shape[0]) {
            for (k in i + 1 until x.shape[0]) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    } else {
        for (i in x.shape[0] - 1 downTo 0) {
            for (k in i - 1 downTo 0) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    }
    return x
}

private fun solveTriangleComplexFloat(a: MultiArray<ComplexFloat, D2>, b: MultiArray<ComplexFloat, D2>, isLowerTriangle: Boolean = true): D2Array<ComplexFloat> {
    require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
    requireSquare(a)

    val x = b.deepCopy() as D2Array<ComplexFloat>
    for (i in 0 until x.shape[0]) {
        for (j in 0 until x.shape[1]) {
            x[i, j] /= a[i, i]
        }
    }
    if (isLowerTriangle) {
        for (i in 0 until x.shape[0]) {
            for (k in i + 1 until x.shape[0]) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    } else {
        for (i in x.shape[0] - 1 downTo 0) {
            for (k in i - 1 downTo 0) {
                for (j in 0 until x.shape[1]) {
                    x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                }
            }
        }
    }
    return x
}