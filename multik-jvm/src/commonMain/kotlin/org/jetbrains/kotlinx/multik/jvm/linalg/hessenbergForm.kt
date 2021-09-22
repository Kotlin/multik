package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * computes Q, H matrices that
 *
 * a = Q * H * Q.H
 *
 * Q is unitary: Q * Q.H = Id
 *
 * H has all zeros below main subdiagonal:
 *
 *  [#, #, #, #]
 *
 *  [#, #, #, #]
 *
 *  [0, #, #, #]
 *
 *  [0, 0, #, #]
 *
 */
internal fun upperHessenberg(a: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    var id = mk.identity<ComplexDouble>(a.shape[0])



    var ans = a.deepCopy() as D2Array<ComplexDouble>

    for (i in 1 until ans.shape[0] - 1) {
        val (tau, v) = householderTransformComplexDouble(ans[i..ans.shape[0], (i - 1)..ans.shape[1]])


        var submatrix = ans[i..ans.shape[0], (i - 1)..ans.shape[1]]
        submatrix = applyHouseholderComplexDouble(submatrix, tau, v)
        //copy
        for (i1 in i until ans.shape[0]) {
            for (j1 in i - 1 until ans.shape[1]) {
                ans[i1, j1] = submatrix[i1 - i, j1 - (i - 1)]
            }
        }

        ans = ans.conjTranspose()

        submatrix = ans[i..ans.shape[0], 0..ans.shape[1]].deepCopy()
        submatrix = applyHouseholderComplexDouble(submatrix, tau, v)

        //copy
        for (i1 in i until ans.shape[0]) {
            for (j1 in 0 until ans.shape[1]) {
                ans[i1, j1] = submatrix[i1 - i, j1]
            }
        }

        ans = ans.conjTranspose()


        submatrix = applyHouseholderComplexDouble(id[i..id.shape[0], 0..id.shape[1]], tau, v)
        for (i1 in i until id.shape[0]) {
            for (j1 in 0 until id.shape[1]) {
                id[i1, j1] = submatrix[i1 - i, j1]
            }
        }
    }
    id = id.conjTranspose()

    // cleaning subdiagonal
    for (i in 2 until ans.shape[0]) {
        for (j in 0 until i - 1) {
            ans[i, j] = ComplexDouble.zero;
        }
    }


    return Pair(id, ans)
}