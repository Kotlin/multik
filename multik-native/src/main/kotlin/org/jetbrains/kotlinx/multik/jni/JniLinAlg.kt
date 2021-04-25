/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.api.LinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object NativeLinAlg : LinAlg {

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
    }

    //TODO (Double and Number type)
    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[1] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
        }

        return if (b.dim.d == 2) {
            val shape = intArrayOf(a.shape[0], b.shape[1])
            when (a.dtype) {
                DataType.FloatDataType -> {
                    val c = FloatArray(shape[0] * shape[1])
                    JniLinAlg.dot(a.data.getFloatArray(), shape[0], shape[1], b.data.getFloatArray(), a.shape[1], c)
                    D2Array<Float>(MemoryViewFloatArray(c), 0, shape, dtype = DataType.FloatDataType, dim = D2)
                }
                DataType.DoubleDataType -> {
                    val c = DoubleArray(shape[0] * shape[1])
                    JniLinAlg.dot(a.data.getDoubleArray(), shape[0], shape[1], b.data.getDoubleArray(), a.shape[1], c)
                    D2Array<Double>(MemoryViewDoubleArray(c), 0, shape, dtype = DataType.DoubleDataType, dim = D2)
                }
                else -> throw UnsupportedOperationException()
            } as NDArray<T, D>
        } else {
            val shape = intArrayOf(a.shape[0])
            when (a.dtype) {
                DataType.FloatDataType -> {
                    val c = FloatArray(shape[0])
                    JniLinAlg.dot(a.data.getFloatArray(), a.shape[0], a.shape[1], b.data.getFloatArray(), c)
                    D1Array<Float>(MemoryViewFloatArray(c), 0, shape, dtype = a.dtype, dim = D1)
                }
                DataType.DoubleDataType -> {
                    val c = DoubleArray(shape[0])
                    JniLinAlg.dot(a.data.getDoubleArray(), a.shape[0], a.shape[1], b.data.getDoubleArray(), c)
                    D1Array<Double>(MemoryViewDoubleArray(c), 0, shape, dtype = a.dtype, dim = D1)
                }
                else -> throw UnsupportedOperationException()
            } as NDArray<T, D>
        }
    }

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        TODO("Not yet implemented")
    }
}

private object JniLinAlg {
    external fun <T : Number> pow(mat: FloatArray, n: Int, result: FloatArray): Unit
    external fun <T : Number> pow(mat: DoubleArray, n: Int, result: DoubleArray): Unit
    external fun <T : Number> norm(mat: FloatArray, p: Int): Double
    external fun <T : Number> norm(mat: DoubleArray, p: Int): Double
    external fun dot(a: FloatArray, m: Int, n: Int, b: FloatArray, k: Int, c: FloatArray): Unit
    external fun dot(a: DoubleArray, m: Int, n: Int, b: DoubleArray, k: Int, c: DoubleArray): Unit
    external fun dot(a: FloatArray, m: Int, n: Int, b: FloatArray, c: FloatArray): Unit
    external fun dot(a: DoubleArray, m: Int, n: Int, b: DoubleArray, c: DoubleArray): Unit
}