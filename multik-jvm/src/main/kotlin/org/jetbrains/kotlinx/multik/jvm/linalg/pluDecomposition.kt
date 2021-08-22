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
import kotlin.math.abs
import kotlin.math.min

internal fun plu(a: MultiArray<Double, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> {
    val _a: D2Array<Double> = a.map { it }
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    pluDecompositionInplace(_a, perm)
    // since previous call _a contains answer

    val L = mk.empty<Double, D2>(_a.shape[0], min(_a.shape[0], _a.shape[1]))
    val U = mk.empty<Double, D2>(min(_a.shape[0], _a.shape[1]), _a.shape[1])

    fillLowerMatrix(L, _a, 1.0, 0.0)

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

internal fun <T : Complex> pluC(a: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> {
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

    fillLowerMatrix(L, _a, ComplexDouble.one, ComplexDouble.zero)

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

    fillLowerMatrix(L, _a, ComplexFloat.one, ComplexFloat.zero)

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

internal fun <T> pluCompressed(a: MultiArray<T, D2>): Triple<D1Array<Int>, D2Array<T>, D2Array<T>> {
    val _a = a.deepCopy()
    val perm = mk.empty<Int, D1>(min(_a.shape[0], _a.shape[1]))

    val L = mk.empty<T, D2>(intArrayOf(_a.shape[0], min(_a.shape[0], _a.shape[1])), a.dtype)
    val U = mk.empty<T, D2>(intArrayOf(min(_a.shape[0], _a.shape[1]), _a.shape[1]), a.dtype)

    when(_a.dtype) {
        DataType.DoubleDataType -> {
            pluDecompositionInplace(_a as D2Array<Double>, perm)
            fillLowerMatrix(L as D2Array<Double>, _a, 1.0, 0.0)
        }
        DataType.FloatDataType -> {
            pluDecompositionInplaceF(_a as D2Array<Float>, perm)
            fillLowerMatrix(L as D2Array<Float>, _a, 1f, 0f)
        }
        DataType.ComplexDoubleDataType -> {
            pluDecompositionInplaceComplexDouble(_a as D2Array<ComplexDouble>, perm)
            fillLowerMatrix(L as D2Array<ComplexDouble>, _a, ComplexDouble.one, ComplexDouble.zero)
        }
        DataType.ComplexFloatDataType -> {
            pluDecompositionInplaceComplexFloat(_a as D2Array<ComplexFloat>, perm)
            fillLowerMatrix(L as D2Array<ComplexFloat>, _a, ComplexFloat.one, ComplexFloat.zero)
        }
        else -> throw UnsupportedOperationException()
    }
    // since previous call _a contains answer

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm, L, U)
}

/**
 *
 */
private fun <T> fillLowerMatrix(L: D2Array<T>, a: D2Array<T>, one: T, zero: T) {
    for (i in 0 until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = when {
                j < i -> a[i, j]
                i == j -> one
                else -> zero
            }
        }
    }
}

/**
 * Solve inplace lower triangle system
 * Solves ax=b equation where [a] lower triangular matrix with units on diagonal,
 * rewrite [b] with solution
 *
 * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
 * it is a contract of this method having no such checks
 *
 * lapack: dtrsm can do it (and have some extra options)
 */
private inline fun solveLowerTriangleSystem(sizeA: Int, sizeX: Int, action: (Int, Int, Int) -> Unit) {
    for (i in 0 until sizeA) {
        for (k in i + 1 until sizeA) {
            for (j in 0 until sizeX) {
                action(i, k, j)
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
    swapLines(rowPerm, from2 = n1, to2 = a.shape[1], swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })

    // change a12
    val aSlice = a[0..n1, 0..n1]
    val x = a[0..n1, n1..(n1 + n2)] as D2Array<Double>
    solveLowerTriangleSystem(aSlice.shape[1], x.shape[1]) { i, k, j -> x[k, j] -= aSlice[k, i] * x[i, j] }

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
    swapLines(rowPerm, n1, to2 = n1,
        swap = { i, j -> a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] } })
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

    swapLines(rowPerm, from2 = n1, to2 = a.shape[1], swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })

    val aSlice = a[0..n1, 0..n1]
    val x = a[0..n1, n1..(n1 + n2)] as D2Array<Float>
    solveLowerTriangleSystem(aSlice.shape[1], x.shape[1]) { i, k, j -> x[k, j] -= aSlice[k, i] * x[i, j] }

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

    swapLines(rowPerm, n1, to2 = n1, swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })
}

private fun pluDecompositionInplaceComplexDouble(a: D2Array<ComplexDouble>, rowPerm: D1Array<Int>) {
    val n1 = min(a.shape[1], a.shape[0]) / 2
    val n2 = a.shape[1] - n1

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

    pluDecompositionInplaceComplexDouble(
        a[0..a.shape[0], 0..n1] as D2Array<ComplexDouble>,
        rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>
    )

    swapLines(rowPerm, from2 = n1, to2 = a.shape[1], swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })

    val aSlice = a[0..n1, 0..n1]
    val x = a[0..n1, n1..(n1 + n2)] as D2Array<ComplexDouble>
    solveLowerTriangleSystem(aSlice.shape[1], x.shape[1]) { i, k, j -> x[k, j] -= aSlice[k, i] * x[i, j] }

    val update = dotMatrixComplex(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)]) // TODO
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    pluDecompositionInplaceComplexDouble(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<ComplexDouble>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    swapLines(rowPerm, n1, to2 = n1, swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })
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

    pluDecompositionInplaceComplexFloat(
        a[0..a.shape[0], 0..n1] as D2Array<ComplexFloat>,
        rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>
    )

    swapLines(rowPerm, from2 = n1, to2 = a.shape[1], swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })

    val aSlice = a[0..n1, 0..n1]
    val x = a[0..n1, n1..(n1 + n2)] as D2Array<ComplexFloat>
    solveLowerTriangleSystem(aSlice.shape[1], x.shape[1]) { i, k, j -> x[k, j] -= aSlice[k, i] * x[i, j] }

    val update = dotMatrixComplex(a[n1..a.shape[0], 0..n1], a[0..n1, n1..(n1 + n2)])
    for (i in n1 until a.shape[0]) {
        for (j in n1 until n1 + n2) {
            a[i, j] -= update[i - n1, j - n1]
        }
    }

    pluDecompositionInplaceComplexFloat(
        a[n1..a.shape[0], n1..a.shape[1]] as D2Array<ComplexFloat>,
        rowPerm[n1..min(a.shape[0], a.shape[1])] as D1Array<Int>
    )

    swapLines(rowPerm, n1, to2 = n1, swap = { i, j ->
        a[i, j] = a[i + rowPerm[i], j].also { a[i + rowPerm[i], j] = a[i, j] }
    })
}
