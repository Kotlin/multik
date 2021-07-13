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
        if (beta == 0.0) {
            for (i in 0 until c.shape[0]) {
                for (j in 0 until c.shape[1]) {
                    c[i, j] = 0.0 //TODO: *= beta
                }
            }
            return
        }
        for (i in 0 until c.shape[0]) {
            for (j in 0 until c.shape[1]) {
                c[i, j] *= beta
            }
        }
        return
    }

    for (j in 0 until c.shape[1]) {
        if (beta == 0.0) {
            for (i in 0 until c.shape[0]) {
                c[i, j] = 0.0
            }
        } else {
            for (i in 0 until c.shape[0]) {
                c[i, j] *= beta
            }
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
 * @param a has na rows and na columns
 * @param b has na rows and nb columns
 *
 * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
 * it is a contract of this method having no such checks
 *
 * lapack: dtrsm can do it (and have some extra options)
 */
private fun solveTriangleInplace(a: D2Array<Double>, offseta0: Int, offseta1: Int, na: Int, b: D2Array<Double>, offsetb0: Int, offsetb1: Int, nb: Int) {
    for (i in 0 until na) {
        for (k in i+1 until na) {
            for (j in 0 until nb) {
                // array getter have extra two bound checks, maybe better use b.data[...]
                b[k + offsetb0, j + offsetb1] -= a[k + offseta0, i + offseta1] * b[i + offsetb0, j + offsetb1] //TODO: move k + offset0 to prev loop
            }
        }
    }
}


/**
 *
 * computes an LU factorization of a matrix @param a
 * Where u is permutation matrix,
 * l lower triangular matrix with unit diagonal elements
 * u upper triangular matrix
 *
 *
 * lapack: dgetrf2
 */
//TODO: a = slice?
private fun PLUdecomposition2Inplace(a: D2Array<Double>, offseta0: Int, offseta1: Int, ma: Int, na: Int, rowPerm: D1Array<Int>) {
    // this is recursive function, position of current matrix we work with is
    // a[offseta0 until offseta0 + am, offseta1 until offseta1 + an]
    val n1 = min(na, ma) / 2
    val n2 = na - n1

    // the idea of an algorithm is represent matrix a as
    // a = [ a11 a12 ]
    //     [ a21 a22 ]
    // where aij -- block submatrices
    // a11.shape = (n1, n1) (others shapes can be calculated using this information)
    // then recursively apply to [ a11 ] and [ a22 ] parts combining results together
    //                           [ a21 ]

    // corner cases
    if(na == 0 || ma == 0) return
    if (ma == 1) return //because [[1]] * a == a

    if (na == 1) {
        var imax = 0
        var elemmax = a[offseta0, offseta1]
        for (i in 1 until ma) { // TODO: maxBy
            if (abs(a[offseta0 + i, offseta1]) > abs(elemmax)) {
                elemmax = a[offseta0 + i, offseta1]
                imax = i
            }
        }

        if (elemmax != 0.0) {
            // pivoting
            a[offseta0, offseta1] = a[offseta0 + imax, offseta1].also {
                a[offseta0 + imax, offseta1] = a[offseta0, offseta1]
            }
            rowPerm[offseta0] = offseta0 + imax

            for (i in 1 until ma) {
                a[offseta0 + i, offseta1] /= elemmax
            }
        }

        return
    }

    // apply recursively to [ a11 ]
    //                      [ a21 ]
    PLUdecomposition2Inplace(a, offseta0, offseta1, ma, n1, rowPerm)

    // change [ a12 ]
    //        [ a22 ]
    for (i in 0 until min(ma, rowPerm.size - offseta0)) {
        if (rowPerm[i + offseta0] != i + offseta0) {
            for (j in n1 until na) {
                a[offseta0 + i, offseta1 + j] = a[rowPerm[i + offseta0], offseta1 + j].also {
                    a[rowPerm[i + offseta0], offseta1 + j] = a[offseta0 + i, offseta1 + j]
                }
            }
        }
    }

    // adjust a12
    solveTriangleInplace(a, offseta0, offseta1, n1, a, offseta0, offseta1 + n1, n2)

    // update a22
    dotThenPlusInplace(
        a[(offseta0 + n1) .. (offseta0 + n1) + ma - n1, (offseta1 + 0)  .. (offseta1 + 0)  + n1] as D2Array<Double>,
        a[(offseta0 + 0)  .. (offseta0 + 0)  + n1     , (offseta1 + n1) .. (offseta1 + n1) + n2] as D2Array<Double>,
        a[(offseta0 + n1) .. (offseta0 + n1) + ma - n1, (offseta1 + n1) .. (offseta1 + n1) + n2] as D2Array<Double>,
        -1.0,
        1.0
    )

    // factor a22
    PLUdecomposition2Inplace(a, offseta0 + n1, offseta1 + n1, ma - n1, n2, rowPerm)

    // apply rowPerm to a21
    for (i in n1 until ma) {
        if(offseta0 + i >= rowPerm.size) {
            break
        }
        if (rowPerm[offseta0 + i] != offseta0 + i) {
            for (j in 0 until n1) {
                a[offseta0 + i, offseta1 + j] = a[rowPerm[offseta0 + i], offseta1 + j].also {
                    a[rowPerm[offseta0 + i], offseta1 + j] = a[offseta0 + i, offseta1 + j]
                }
            }
        }
    }

}

fun PLU(a: D2Array<Double>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> {
    val _a = a.deepCopy()
    val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }
    for (i in perm.indices) perm[i] = i

    PLUdecomposition2Inplace(_a, 0, 0, _a.shape[0], _a.shape[1], perm)
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
        if(perm[i] != i) {
            P[i] = P[perm[i]].deepCopy().also { P[perm[i]] = P[i].deepCopy() }
        }
    }
    return Triple(P, L, U)
}

fun PLUCompressed(a: D2Array<Double>): Triple<D1Array<Int>, D2Array<Double>, D2Array<Double>> {
    val _a = a.deepCopy()
    val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }
    for (i in perm.indices) perm[i] = i

    PLUdecomposition2Inplace(_a, 0, 0, _a.shape[0], _a.shape[1], perm)
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


    val P = mk.identity<Double>(_a.shape[0])

    return Triple(perm.deepCopy(), L, U)
}


