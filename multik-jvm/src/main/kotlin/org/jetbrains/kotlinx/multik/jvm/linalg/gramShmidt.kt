package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.abs
import kotlin.math.sqrt


internal fun gramShmidtComplexDouble(a: MultiArray<ComplexDouble, D2>): D2Array<ComplexDouble> {
    val precision = 1e-16
    val ans: D2Array<ComplexDouble> = (a.deepCopy() as D2Array<ComplexDouble>)

    //normalize all vwctors to have unti length
    for (j in 0 until ans.shape[1]) {
        var norm = 0.0
        for (i in 0 until ans.shape[0]) {
            norm += (ans[i, j] * ans[i, j].conjugate()).re
        }
        norm = sqrt(abs(norm))
        if (norm < precision) {
            continue;
        }
        for (i in 0 until ans.shape[0]) {
            ans[i, j] = ans[i, j] / norm
        }
    }

    for (curcol in 1 until ans.shape[1]) {
        // make curcol vector orthogonal to previous vectors
        for (j in 0 until curcol) {
            var scalProd = ComplexDouble.zero
            for (i in 0 until ans.shape[0]) {
                scalProd += ans[i, j] * ans[i, curcol].conjugate()
            }
            for (i in 0 until ans.shape[0]) {
                ans[i, curcol] = ans[i, curcol] - scalProd.conjugate() * ans[i, j]
            }
        }
        // curcol vector normalization
        var norm = 0.0
        for (i in 0 until ans.shape[0]) {
            norm += (ans[i, curcol] * ans[i, curcol].conjugate()).re
        }
        norm = sqrt(abs(norm))
        if(norm < precision) {
            continue
        }
        for (i in 0 until ans.shape[0]) {
            ans[i, curcol] = ans[i, curcol] / norm
        }
    }

    return ans
}