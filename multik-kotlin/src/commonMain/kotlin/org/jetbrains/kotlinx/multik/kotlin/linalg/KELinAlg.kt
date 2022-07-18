/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.kotlin.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object KELinAlg : LinAlg {

    override val linAlgEx: LinAlgEx
        get() = KELinAlgEx

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        if (n == 0) return mk.identity(mat.shape[0], mat.dtype)

        return if (n % 2 == 0) {
            val tmp = pow(mat, n / 2)
            dot(tmp, tmp)
        } else {
            dot(mat, pow(mat, n - 1))
        }
    }

//    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
//        require(p > 0) { "Power $p must be positive" }
//
//        return when (mat.dtype) {
//            DataType.DoubleDataType -> {
//                norm(mat.data.getDoubleArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
//            }
//            DataType.FloatDataType -> {
//                norm(mat.data.getFloatArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
//            }
//            else -> {
//                norm(mat.data, mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
//            }
//        }
//    }
}
