


package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*

internal fun deepCopyMatrixTmp(a: MultiArray<ComplexDouble, D2>): D2Array<ComplexDouble> {
    val ans = mk.empty<ComplexDouble, D2>(a.shape[0], a.shape[1])
    for (i in 0 until a.shape[0]) {
        for (j in 0 until a.shape[1]) {
            ans[i, j] = a[i, j]
        }
    }
    return ans;
}


internal fun deepCopyVectorTmp(a: NDArray<ComplexDouble, D1>): D1Array<ComplexDouble> {
    val ans = mk.empty<ComplexDouble, D1>(a.shape[0])
    for (i in 0 until a.shape[0]) {
            ans[i] = a[i]
    }
    return ans;
}
