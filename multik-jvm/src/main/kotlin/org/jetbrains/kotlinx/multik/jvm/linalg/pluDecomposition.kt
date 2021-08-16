package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import java.lang.UnsupportedOperationException
import kotlin.math.abs
import kotlin.math.min

//-------------------Double and Float case-----------------
/**
 * Solves ax=b equation where @param a lower triangular matrix with units on diagonal,
 * rewrite @param b with solution
 *
 * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
 * it is a contract of this method having no such checks
 *
 * lapack: dtrsm can do it (and have some extra options)
 */
private fun solveLowerTriangleSystemInplace(a: MultiArray<Double, D2>, b: D2Array<Double>) {
    for (i in 0 until a.shape[1]) {
        for (k in i + 1 until a.shape[1]) {
            for (j in 0 until b.shape[1]) {
                b[k, j] -= a[k, i] * b[i, j]
            }
        }
    }
}

private fun solveLowerTriangleSystemInplaceF(a: MultiArray<Float, D2>, b: D2Array<Float>) {
    for (i in 0 until a.shape[1]) {
        for (k in i + 1 until a.shape[1]) {
            for (j in 0 until b.shape[1]) {
                b[k, j] -= a[k, i] * b[i, j]
            }
        }
    }
}

/**
 *
 * computes an PLU factorization of a matrix @param a
 * Where p is permutation,
 * L lower triangular matrix with unit diagonal elements
 * U upper triangular matrix
 *
 * Stores result in a in such way:
 * a[i, <i] contains L matrix (without units on main diagonal)
 * a[i, >=i] contains U matrix
 *
 * rowPerm is permutation such that: rowPerm ⚬ a = LU, so
 * a = rowPerm^(-1) ⚬ (LU)
 *
 * rowPerm ⚬ array is defined in following way:
 * for (i in rowPerm.indices) {
 *     swap(array[i], array[i + rowPerm[i]])
 * }
 * rowPerm^(-1) ⚬ array is defined as:
 * for (i in rowPerm.indices.reversed()) {
 *     swap(array[i], array[i + rowPerm[i]])
 * }
 *
 * it's not hard to proof that
 * rowPerm ⚬ (rowPerm^(-1) ⚬ array) = rowPerm^(-1) ⚬ (rowPerm ⚬ array) = array
 *
 * lapack: dgetrf2
 */
private fun pluDecompositionInplace(a: D2Array<Double>, rowPerm: D1Array<Int>) {
    // this is recursive function, position of current matrix we work with is
    // a[0 until am, 0 until an]
    val n1 = min(a.shape[1], a.shape[0]) / 2
    val n2 = a.shape[1] - n1

    // the idea of an algorithm is represent matrix a as
    // a = [ a11 a12 ]
    //     [ a21 a22 ]
    // where aij -- block submatrices
    // a11.shape = (n1, n1) (others shapes can be calculated using this information)
    // then recursively apply to [ a11 ] and [ a22 ] parts combining results together
    //                           [ a21 ]

    // corner cases
    if (a.shape[1] == 0 || a.shape[0] == 0) return
    if (a.shape[0] == 1) return //because [[1]] * a == a

    if (a.shape[1] == 1) {
        var imax = 0
        var elemmax = a[0, 0]
        for (i in 1 until a.shape[0]) {
            if (abs(a[i, 0]) > abs(elemmax)) {
                elemmax = a[i, 0]
                imax = i
            }
        }

        if (elemmax != 0.0) {
            // pivoting
            a[0, 0] = a[imax, 0].also { a[imax, 0] = a[0, 0] }
            rowPerm[0] = imax

            for (i in 1 until a.shape[0]) {
                a[i, 0] /= elemmax
            }
        }

        return
    }

    // apply recursively to [ a11 ]
    //                      [ a21 ]
    pluDecompositionInplace(a[0..a.shape[0], 0..n1] as D2Array<Double>, rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>)


    // change [ a12 ]
    //        [ a22 ]
    a.swapLines(rowPerm, from2 = n1, to2 = a.shape[1])

    // change a12
    solveLowerTriangleSystemInplace(a[0..n1, 0..n1], a[0..n1, n1..(n1 + n2)] as D2Array<Double>)

    // update a22
    val update = dotMatrix(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)]) // TODO
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    // factor a22
    pluDecompositionInplace(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<Double>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    // apply rowPerm to a21
    a.swapLines(rowPerm, n1, to2 = n1)
}

private fun pluDecompositionInplaceF(a: D2Array<Float>, rowPerm: D1Array<Int>) {
    val n1 = min(a.shape[1], a.shape[0]) / 2
    val n2 = a.shape[1] - n1

    if (a.shape[1] == 0 || a.shape[0] == 0) return
    if (a.shape[0] == 1) return

    if (a.shape[1] == 1) {
        var imax = 0
        var elemmax = a[0, 0]
        for (i in 1 until a.shape[0]) {
            if (abs(a[i, 0]) > abs(elemmax)) {
                elemmax = a[i, 0]
                imax = i
            }
        }

        if (elemmax != 0f) {
            a[0, 0] = a[imax, 0].also { a[imax, 0] = a[0, 0] }
            rowPerm[0] = imax

            for (i in 1 until a.shape[0]) {
                a[i, 0] /= elemmax
            }
        }

        return
    }

    pluDecompositionInplaceF(a[0..a.shape[0], 0..n1] as D2Array<Float>, rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>)


    a.swapLines(rowPerm, from2 = n1, to2 = a.shape[1])

    solveLowerTriangleSystemInplaceF(a[0..n1, 0..n1], a[0..n1, n1..(n1 + n2)] as D2Array<Float>)

    val update = dotMatrix(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)]) // TODO
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    pluDecompositionInplaceF(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<Float>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    a.swapLines(rowPerm, n1, to2 = n1)
}

private fun <T: Number> D2Array<T>.swapLines(
    rowPerm: D1Array<Int>, from1: Int = 0, to1: Int = rowPerm.size, from2: Int = 0, to2: Int
) {
    for (i in from1 until to1) {
        if (rowPerm[i] != 0) {
            for (j in from2 until to2) {
                this[i, j] = this[i + rowPerm[i], j].also { this[i + rowPerm[i], j] = this[i, j] }
            }
        }
    }
}


internal fun plu(a: MultiArray<Double, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> {
    val _a: D2Array<Double> = a.map { it }
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplace(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<Double, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<Double, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> 1.0
                else -> 0.0
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    val P = mk.identity<Double>(_a.shape[0])

    for (i in perm.indices.reversed()) {
        if (perm[i] != 0) {
            P[i] = P[i + perm[i]].deepCopy().also { P[i + perm[i]] = P[i].deepCopy() }
        }
    }
    return Triple(P, L, U)
}

internal fun pluCompressed(a: MultiArray<Double, D2>): Triple<D1Array<Int>, D2Array<Double>, D2Array<Double>> {
    val _a = a.deepCopy() as NDArray<Double, D2>
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplace(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<Double, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<Double, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> 1.0
                else -> 0.0
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm, L, U)
}

internal fun pluCompressedF(a: MultiArray<Float, D2>): Triple<D1Array<Int>, D2Array<Float>, D2Array<Float>> {
    val _a = a.deepCopy()
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplaceF(_a as D2Array<Float>, perm)
    // since previous call _a contains answer

    val L = mk.empty<Float, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<Float, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> 1f
                else -> 0f
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm, L, U)
}

//----------Complex Case----------------

/**
 * Solves ax=b equation where @param a lower triangular matrix with units on diagonal,
 * rewrite @param b with solution
 *
 * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
 * it is a contract of this method having no such checks
 *
 * lapack: dtrsm can do it (and have some extra options)
 */
private fun solveLowerTriangleSystemInplaceComplexDouble(a: MultiArray<ComplexDouble, D2>, b: D2Array<ComplexDouble>) {
    for (i in 0 until a.shape[1]) {
        for (k in i + 1 until a.shape[1]) {
            for (j in 0 until b.shape[1]) {
                b[k, j] -= a[k, i] * b[i, j]
            }
        }
    }
}

private fun solveLowerTriangleSystemInplaceComplexFloat(a: MultiArray<ComplexFloat, D2>, b: D2Array<ComplexFloat>) {
    for (i in 0 until a.shape[1]) {
        for (k in i + 1 until a.shape[1]) {
            for (j in 0 until b.shape[1]) {
                b[k, j] -= a[k, i] * b[i, j]
            }
        }
    }
}

/**
 *
 * computes an PLU factorization of a matrix @param a
 * Where p is permutation,
 * L lower triangular matrix with unit diagonal elements
 * U upper triangular matrix
 *
 * Stores result in a in such way:
 * a[i, <i] contains L matrix (without units on main diagonal)
 * a[i, >=i] contains U matrix
 *
 * rowPerm is permutation such that: rowPerm ⚬ a = LU, so
 * a = rowPerm^(-1) ⚬ (LU)
 *
 * rowPerm ⚬ array is defined in following way:
 * for (i in rowPerm.indices) {
 *     swap(array[i], array[i + rowPerm[i]])
 * }
 * rowPerm^(-1) ⚬ array is defined as:
 * for (i in rowPerm.indices.reversed()) {
 *     swap(array[i], array[i + rowPerm[i]])
 * }
 *
 * it's not hard to proof that
 * rowPerm ⚬ (rowPerm^(-1) ⚬ array) = rowPerm^(-1) ⚬ (rowPerm ⚬ array) = array
 *
 * lapack: dgetrf2
 */
private fun pluDecompositionInplaceComplexDouble(a: D2Array<ComplexDouble>, rowPerm: D1Array<Int>) {
    // this is recursive function, position of current matrix we work with is
    // a[0 until am, 0 until an]
    val n1 = min(a.shape[1], a.shape[0]) / 2
    val n2 = a.shape[1] - n1

    // the idea of an algorithm is represent matrix a as
    // a = [ a11 a12 ]
    //     [ a21 a22 ]
    // where aij -- block submatrices
    // a11.shape = (n1, n1) (others shapes can be calculated using this information)
    // then recursively apply to [ a11 ] and [ a22 ] parts combining results together
    //                           [ a21 ]

    // corner cases
    if (a.shape[1] == 0 || a.shape[0] == 0) return
    if (a.shape[0] == 1) return //because [[1]] * a == a

    if (a.shape[1] == 1) {
        var imax = 0
        var elemmax = a[0, 0]
        for (i in 1 until a.shape[0]) {
            if (a[i, 0].abs() > elemmax.abs()) {
                elemmax = a[i, 0]
                imax = i
            }
        }

        if (elemmax != ComplexDouble.zero) {
            // pivoting
            a[0, 0] = a[imax, 0].also { a[imax, 0] = a[0, 0] }
            rowPerm[0] = imax

            for (i in 1 until a.shape[0]) {
                a[i, 0] /= elemmax
            }
        }

        return
    }

    // apply recursively to [ a11 ]
    //                      [ a21 ]
    pluDecompositionInplaceComplexDouble(a[0..a.shape[0], 0..n1] as D2Array<ComplexDouble>, rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>)


    // change [ a12 ]
    //        [ a22 ]
    a.swapLinesComplex(rowPerm, from2 = n1, to2 = a.shape[1])

    // change a12
    solveLowerTriangleSystemInplaceComplexDouble(a[0..n1, 0..n1], a[0..n1, n1..(n1 + n2)] as D2Array<ComplexDouble>)

    // update a22
    val update = dotMatrixComplex(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)]) // TODO
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    // factor a22
    pluDecompositionInplaceComplexDouble(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<ComplexDouble>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    // apply rowPerm to a21
    a.swapLinesComplex(rowPerm, n1, to2 = n1)
}

private fun pluDecompositionInplaceComplexFloat(a: D2Array<ComplexFloat>, rowPerm: D1Array<Int>) {
    val n1 = min(a.shape[1], a.shape[0]) / 2
    val n2 = a.shape[1] - n1

    if (a.shape[1] == 0 || a.shape[0] == 0) return
    if (a.shape[0] == 1) return

    if (a.shape[1] == 1) {
        var imax = 0
        var elemmax = a[0, 0]
        for (i in 1 until a.shape[0]) {
            if (a[i, 0].abs() > elemmax.abs()) {
                elemmax = a[i, 0]
                imax = i
            }
        }

        if (elemmax != ComplexFloat.zero) {
            a[0, 0] = a[imax, 0].also { a[imax, 0] = a[0, 0] }
            rowPerm[0] = imax

            for (i in 1 until a.shape[0]) {
                a[i, 0] /= elemmax
            }
        }

        return
    }

    pluDecompositionInplaceComplexFloat(a[0..a.shape[0], 0..n1] as D2Array<ComplexFloat>, rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>)


    a.swapLinesComplex(rowPerm, from2 = n1, to2 = a.shape[1])

    solveLowerTriangleSystemInplaceComplexFloat(a[0..n1, 0..n1], a[0..n1, n1..(n1 + n2)] as D2Array<ComplexFloat>)

    val update = dotMatrixComplex(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)]) // TODO
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    pluDecompositionInplaceComplexFloat(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<ComplexFloat>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    a.swapLinesComplex(rowPerm, n1, to2 = n1)
}

private fun <T: Complex> D2Array<T>.swapLinesComplex(
    rowPerm: D1Array<Int>, from1: Int = 0, to1: Int = rowPerm.size, from2: Int = 0, to2: Int
) {
    for (i in from1 until to1) {
        if (rowPerm[i] != 0) {
            for (j in from2 until to2) {
                this[i, j] = this[i + rowPerm[i], j].also { this[i + rowPerm[i], j] = this[i, j] }
            }
        }
    }
}

internal fun <T: Complex> pluC(a: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> {
    return when (a.dtype) {
        DataType.ComplexDoubleDataType -> {
            a as MultiArray<ComplexDouble, D2>
            val (P, L, U) = pluComplexDouble(a)
            Triple(P, L, U)
        }
        DataType.ComplexFloatDataType -> {
            a as MultiArray<ComplexFloat, D2>
            val (P, L, U) = pluComplexFloat(a)
            Triple(P, L, U)
        }
        else -> throw UnsupportedOperationException("matrix should be complex")
        
    } as Triple<D2Array<T>, D2Array<T>, D2Array<T>>
}

internal fun pluComplexDouble(a: MultiArray<ComplexDouble, D2>): Triple<D2Array<ComplexDouble>, D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val _a: D2Array<ComplexDouble> = a.deepCopy() as D2Array<ComplexDouble>
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplaceComplexDouble(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<ComplexDouble, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<ComplexDouble, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> ComplexDouble.one
                else -> ComplexDouble.zero
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    val P = mk.identity<ComplexDouble>(_a.shape[0])

    for (i in perm.indices.reversed()) {
        if (perm[i] != 0) {
            P[i] = P[i + perm[i]].deepCopy().also { P[i + perm[i]] = P[i].deepCopy() }
        }
    }
    return Triple(P, L, U)
}


internal fun pluComplexFloat(a: MultiArray<ComplexFloat, D2>): Triple<D2Array<ComplexFloat>, D2Array<ComplexFloat>, D2Array<ComplexFloat>> {
    val _a: D2Array<ComplexFloat> = a.deepCopy() as D2Array<ComplexFloat>
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplaceComplexFloat(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<ComplexFloat, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<ComplexFloat, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> ComplexFloat.one
                else -> ComplexFloat.zero
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    val P = mk.identity<ComplexFloat>(_a.shape[0])

    for (i in perm.indices.reversed()) {
        if (perm[i] != 0) {
            P[i] = P[i + perm[i]].deepCopy().also { P[i + perm[i]] = P[i].deepCopy() }
        }
    }
    return Triple(P, L, U)
}


internal fun pluCompressedComplexDouble(a: MultiArray<ComplexDouble, D2>): Triple<D1Array<Int>, D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val _a = a.deepCopy() as NDArray<ComplexDouble, D2>
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplaceComplexDouble(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<ComplexDouble, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<ComplexDouble, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> ComplexDouble.one
                else -> ComplexDouble.zero
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm, L, U)
}

internal fun pluCompressedComplexFloat(a: MultiArray<ComplexFloat, D2>): Triple<D1Array<Int>, D2Array<ComplexFloat>, D2Array<ComplexFloat>> {
    val _a = a.deepCopy()
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplaceComplexFloat(_a as D2Array<ComplexFloat>, perm)
    // since previous call _a contains answer

    val L = mk.empty<ComplexFloat, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<ComplexFloat, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> _a[i, j]
                i == j -> ComplexFloat.one
                else -> ComplexFloat.zero
            }
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm, L, U)
}
