package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.sqrt



internal fun upperHessenberg(a: NDArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    var id = mk.empty<ComplexDouble, D2>(a.shape[0], a.shape[0])
    for (i in 0 until a.shape[0]) {
        id[i, i] = ComplexDouble(1.0, 0.0)
    }



    var ans = deepCopyMatrixTmp(a)

    for (i in 1 until ans.shape[0] - 1) {
        val (tau, v) = householderTransformComplexDouble(ans[i..ans.shape[0], (i - 1)..ans.shape[1]])


        var submatrix = deepCopyMatrixTmp(ans[i..ans.shape[0], (i - 1)..ans.shape[1]])
        submatrix = applyHouseholderComplexDouble(submatrix, tau, v)
        //copy
        for (i1 in i until ans.shape[0]) {
            for (j1 in i - 1 until ans.shape[1]) {
                ans[i1, j1] = submatrix[i1 - i, j1 - (i - 1)]
            }
        }

        ans = ans.conjTranspose()

        submatrix = deepCopyMatrixTmp(ans[i..ans.shape[0], 0..ans.shape[1]])
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
        //println("-----------------------------debug ${i}------------------------------------")
//        //println(ans)
        //println(tempDot(tempDot(id.conjTranspose(), ans), id))
        //println("---------------------------------------------------------------------------")

    }
    id = id.conjTranspose()

    return Pair(id, ans)
}