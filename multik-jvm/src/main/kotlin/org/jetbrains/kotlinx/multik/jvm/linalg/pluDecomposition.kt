package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.abs
import kotlin.math.min

// TODO
internal fun <T> pluCompressed(mat: MultiArray<T, D2>): Triple<D1Array<Int>, D2Array<T>, D2Array<T>> {
    val (n, m) = mat.shape
    val nm = min(n, m)

    val perm = mk.empty<Int, D1>(nm)
    val L = mk.empty<T, D2>(intArrayOf(n, nm), mat.dtype)
    val U = mk.empty<T, D2>(intArrayOf(nm, m), mat.dtype)

    when(mat.dtype) {
        DataType.DoubleDataType -> {
            pluDecompositionInplace(mat as D2Array<Double>, perm)
            fillLowerMatrix(L as D2Array<Double>, mat, 1.0)
        }
        DataType.FloatDataType -> {
            pluDecompositionInplaceF(mat as D2Array<Float>, perm)
            fillLowerMatrix(L as D2Array<Float>, mat, 1f)
        }
        DataType.ComplexDoubleDataType -> {
            pluDecompositionInplaceComplexDouble(mat as D2Array<ComplexDouble>, perm)
            fillLowerMatrix(L as D2Array<ComplexDouble>, mat, ComplexDouble.one)
        }
        DataType.ComplexFloatDataType -> {
            pluDecompositionInplaceComplexFloat(mat as D2Array<ComplexFloat>, perm)
            fillLowerMatrix(L as D2Array<ComplexFloat>, mat, ComplexFloat.one)
        }
        else -> throw UnsupportedOperationException()
    }

    // todo fillUpperMatrix
    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = mat[i, j]
        }
    }

    return Triple(perm, L, U)
}

// TODO eye
// TODO fillLowerUpperMatrix
private fun <T> fillLowerMatrix(L: D2Array<T>, a: D2Array<T>, one: T) {
    for (i in 0 until L.shape[0]) {
        for (j in 0..i) {
            L[i, j] = when {
                j >= L.shape[1] -> break
                i == j -> one
                else -> a[i, j]
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
    }) // TODO swap rows on matrix

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
