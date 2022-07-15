/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object NativeLinAlg : LinAlg {

    override val linAlgEx: LinAlgEx
        get() = NativeLinAlgEx

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        requireSquare(mat.shape)
        if (n == 0) return mk.identity(mat.shape[0], mat.dtype)
        return if (n % 2 == 0) {
            val tmp = pow(mat, n / 2)
            NativeLinAlgEx.dotMM(tmp, tmp)
        } else {
            NativeLinAlgEx.dotMM(mat, pow(mat, n - 1))
        }
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
    }
}