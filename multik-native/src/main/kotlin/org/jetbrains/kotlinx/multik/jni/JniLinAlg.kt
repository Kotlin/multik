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

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[0] == a.shape[1]) { "Ndarray must be square: a.shape = ${a.shape.joinToString(",", "(", ")")}" }
        require(a.shape[0] == b.shape[0]) { "The first dimensions of the ndarrays a and b must be match: ${a.shape[0]}(a.shape[0]) != ${b.shape[0]}(b.shape[0]" }
        val ret = if (b.consistent) (b as NDArray).clone() else (b as NDArray).deepCopy()
        val nhrs = if (ret.dim.d == 1) ret.shape[0] else ret.shape[1]

        val info: Int = when (a.dtype) {
            DataType.FloatDataType -> JniLinAlg.solve(
                a.shape.first(), nhrs, a.data.getFloatArray().copyOf(), a.strides.first(),
                ret.data.getFloatArray(), ret.strides.first()
            )
            DataType.DoubleDataType -> JniLinAlg.solve(
                a.shape.first(), nhrs, a.data.getDoubleArray().copyOf(), a.strides.first(),
                ret.data.getDoubleArray(), ret.strides.first()
            )
            else -> throw UnsupportedOperationException()
        }
        if (info > 0) {
            throw Exception("The diagonal element of the triangular factor of a is zero, so that A is singular. The solution could not be computed.")
        }

        return ret
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
        require(a.size == b.size) { "Vector sizes don't match: ${a.size} != ${b.size}" }
        return when (a.dtype) {
            DataType.FloatDataType -> {
                JniLinAlg.dot(
                    a.size,
                    a.data.getFloatArray(),
                    a.strides.first(),
                    b.data.getFloatArray(),
                    b.strides.first()
                )
            }
            DataType.DoubleDataType -> {
                JniLinAlg.dot(
                    a.size,
                    a.data.getDoubleArray(),
                    a.strides.first(),
                    b.data.getDoubleArray(),
                    b.strides.first()
                )
            }
            else -> throw UnsupportedOperationException()
        } as T
    }
}

private object JniLinAlg {
    external fun pow(mat: FloatArray, n: Int, result: FloatArray): Unit
    external fun pow(mat: DoubleArray, n: Int, result: DoubleArray): Unit
    external fun norm(mat: FloatArray, p: Int): Double
    external fun norm(mat: DoubleArray, p: Int): Double
    external fun solve(n: Int, nrhs: Int, a: FloatArray, strA: Int, b: FloatArray, strB: Int): Int
    external fun solve(n: Int, nrhs: Int, a: DoubleArray, strA: Int, b: DoubleArray, strB: Int): Int
    external fun dot(a: FloatArray, m: Int, n: Int, b: FloatArray, k: Int, c: FloatArray): Unit
    external fun dot(a: DoubleArray, m: Int, n: Int, b: DoubleArray, k: Int, c: DoubleArray): Unit
    external fun dot(a: FloatArray, m: Int, n: Int, b: FloatArray, c: FloatArray): Unit
    external fun dot(a: DoubleArray, m: Int, n: Int, b: DoubleArray, c: DoubleArray): Unit
    external fun dot(n: Int, a: FloatArray, strideA: Int, b: FloatArray, strideB: Int): Float
    external fun dot(n: Int, a: DoubleArray, strideA: Int, b: DoubleArray, strideB: Int): Double
}