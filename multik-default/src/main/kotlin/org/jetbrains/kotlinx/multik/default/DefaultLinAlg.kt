/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.LinAlg
import org.jetbrains.kotlinx.multik.jni.NativeEngine
import org.jetbrains.kotlinx.multik.jni.NativeLinAlg
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object DefaultLinAlg : LinAlg {
    init {
        NativeEngine
    }

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        return JvmLinAlg.pow(mat, n)
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double = JvmLinAlg.norm(mat, p)

    override fun <T : Number, D: Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        return when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> throw UnsupportedOperationException()
        }
    }

    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        return when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> JvmLinAlg.dot(a, b)
        }
    }

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        return when(a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> JvmLinAlg.dot(a, b)
        }
    }

}