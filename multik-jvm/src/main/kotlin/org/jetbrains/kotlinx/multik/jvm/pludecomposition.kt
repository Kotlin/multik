package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.abs
import kotlin.math.min

/**
 * computes alpha * a * b + beta * c and stores result is c
 * @param alpha scalar
 * @param beta scalar
 * @param a matrix
 * @param b matrix
 * @param c matrix
 *
 * blas / lapack: dgemm
 */
private fun dotThenPlusInplace(a: D2Array<Double>, b: D2Array<Double>, c: D2Array<Double>, alpha: Double, beta: Double) {
    if(c.shape[0] <= 0 || c.shape[1] <= 0) return

    if (alpha == 0.0) {
        for (i in 0 until c.shape[0]) {
            for (j in 0 until c.shape[1]) {
                c[i, j] *= beta
            }
        }
        return
    }

    for (j in 0 until c.shape[1]) {
        for (i in 0 until c.shape[0]) {
            c[i, j] *= beta
        }

        for (l in 0 until a.shape[1]) {
            val temp = alpha * b[l, j]
            for (i in 0 until c.shape[0]) {
                c[i, j] += temp * a[i, l]
            }
        }
    }
}

/**
 * Solves ax=b equation where @param a lower triangular matrix with units on diagonal,
 * rewrite @param b with solution
 *
 * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
 * it is a contract of this method having no such checks
 *
 * lapack: dtrsm can do it (and have some extra options)
 */
private fun solveLowerTriangleSystemInplace(a: D2Array<Double>, b: D2Array<Double>) {
    for (i in 0 until a.shape[1]) {
        for (k in i+1 until a.shape[1]) {
            for (j in 0 until b.shape[1]) {
                // array getter have extra two bound checks, maybe better use b.data[...]
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
private fun PLUdecomposition2Inplace(a: D2Array<Double>, rowPerm: D1Array<Int>) {
    // this is recursive function, position of current matrix we work with is
    // a[0 until am, 0 until an]
    val n1 = min(a.shape[1], a.shape[0] ) / 2
    val n2 = a.shape[1] - n1

    // the idea of an algorithm is represent matrix a as
    // a = [ a11 a12 ]
    //     [ a21 a22 ]
    // where aij -- block submatrices
    // a11.shape = (n1, n1) (others shapes can be calculated using this information)
    // then recursively apply to [ a11 ] and [ a22 ] parts combining results together
    //                           [ a21 ]

    // corner cases
    if(a.shape[1] == 0 || a.shape[0] == 0) return
    if (a.shape[0] == 1) return //because [[1]] * a == a

    if (a.shape[1] == 1) {
        var imax = 0
        var elemmax = a[0, 0]
        for (i in 1 until a.shape[0] ) {
            if (abs(a[i, 0]) > abs(elemmax)) {
                elemmax = a[i, 0]
                imax = i
            }
        }

        if (elemmax != 0.0) {
            // pivoting
            a[0, 0] = a[imax, 0].also {
                a[imax, 0] = a[0, 0]
            }
            rowPerm[0] = imax

            for (i in 1 until a.shape[0] ) {
                a[i, 0] /= elemmax
            }
        }

        return
    }

    // apply recursively to [ a11 ]
    //                      [ a21 ]
    PLUdecomposition2Inplace(a[0..a.shape[0], 0..n1] as D2Array<Double>, rowPerm[0..min(a.shape[0], n1)] as D1Array<Int>)

    // change [ a12 ]
    //        [ a22 ]
    for (i in rowPerm.indices) {
        if (rowPerm[i] != 0) {
            for (j in n1 until a.shape[1]) {
                a[i, j] = a[i + rowPerm[i], j].also {
                    a[i + rowPerm[i], j] = a[i, j]
                }
            }
        }
    }

    // change a12
    solveLowerTriangleSystemInplace(
        a[0 .. n1, 0  .. n1     ] as D2Array<Double>,
        a[0 .. n1, n1 .. n1 + n2] as D2Array<Double>
    )

    // update a22
    dotThenPlusInplace(
        a[n1 .. n1 + a.shape[0] - n1, 0  .. n1     ] as D2Array<Double>,
        a[0  .. n1                  , n1 .. n1 + n2] as D2Array<Double>,
        a[n1 .. n1 + a.shape[0] - n1, n1 .. n1 + n2] as D2Array<Double>,
        -1.0,
        1.0
    )

    // factor a22
    PLUdecomposition2Inplace(a[n1 .. a.shape[0], n1 .. a.shape[1]] as D2Array<Double>,  rowPerm[n1 .. min(a.shape[0], a.shape[1])] as D1Array<Int>)

    // apply rowPerm to a21
    for (i in n1 until a.shape[0] ) {
        if(i >= rowPerm.size) {
            break
        }
        if (rowPerm[i] != 0) {
            for (j in 0 until n1) {
                a[i, j] = a[i + rowPerm[i], j].also {
                    a[i + rowPerm[i], j] = a[i, j]
                }
            }
        }
    }

}


fun PLU(a: D2Array<Double>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> {
    val _a = a.deepCopy()
    val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }


    PLUdecomposition2Inplace(_a, perm)
    // since previous call _a contains answer

    val L = mk.d2array(_a.shape[0], min(_a.shape[0], _a.shape[1])) { 0.0 }
    val U = mk.d2array(min(_a.shape[0], _a.shape[1]), _a.shape[1]) {0.0}

    for (i in 0 until L.shape[1]) {
        L[i, i] = 1.0
        for (j in 0 until i) {
            L[i, j] = _a[i, j]
        }
    }
    for (i in L.shape[1] until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = _a[i, j]
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }


    val P = mk.identity<Double>(_a.shape[0])

    for (i in perm.indices.reversed()) {
        if(perm[i] != 0) {
            P[i] = P[i + perm[i]].deepCopy().also { P[i + perm[i]] = P[i].deepCopy() }
        }
    }
    return Triple(P, L, U)
}

fun PLUCompressed(a: D2Array<Double>): Triple<D1Array<Int>, D2Array<Double>, D2Array<Double>> {
    val _a = a.deepCopy()
    val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }


    PLUdecomposition2Inplace(_a, perm)
    // since previous call _a contains answer

    val L = mk.d2array(_a.shape[0], min(_a.shape[0], _a.shape[1])) {0.0}
    val U = mk.d2array(min(_a.shape[0], _a.shape[1]), _a.shape[1]) {0.0}

    for (i in 0 until L.shape[1]) {
        L[i, i] = 1.0
        for (j in 0 until i) {
            L[i, j] = _a[i, j]
        }
    }
    for (i in L.shape[1] until L.shape[0]) {
        for (j in 0 until L.shape[1]) {
            L[i, j] = _a[i, j]
        }
    }

    for (i in 0 until U.shape[0]) {
        for (j in i until U.shape[1]) {
            U[i, j] = _a[i, j]
        }
    }

    return Triple(perm.deepCopy(), L, U)
}


