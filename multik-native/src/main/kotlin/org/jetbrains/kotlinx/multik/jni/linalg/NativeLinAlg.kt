/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.jni.NativeEngine
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object NativeLinAlg : LinAlg {
    init {
        NativeEngine
    }

    override val linAlgEx: LinAlgEx
        get() = NativeLinAlgEx

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
    }

//    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<T, D2> {
//        require(mat.shape[0] == mat.shape[1]) { "Ndarray must be square: mat.shape = ${mat.shape.joinToString(",", "(", ")")}"}
//
//        val ret = if (mat.consistent) (mat.clone() as NDArray) else (mat.deepCopy() as NDArray)
//        val info: Int = when (mat.dtype) {
//            DataType.FloatDataType -> JniLinAlg.inv(ret.shape.first(), ret.data.getFloatArray(), ret.strides.first())
//            DataType.DoubleDataType -> JniLinAlg.inv(ret.shape.first(), ret.data.getDoubleArray(), ret.strides.first())
//            else -> throw UnsupportedOperationException()
//        }
//
//        when {
//            info < 0 -> throw IllegalArgumentException("${-info} argument had illegal value. ")
//            info > 0 -> throw Exception("U($info, $info) is exactly zero. Matrix is singular and its inverse could not be computed")
//        }
//
//        return ret
//    }

//    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
//        require(a.shape[0] == a.shape[1]) { "Ndarray must be square: a.shape = ${a.shape.joinToString(",", "(", ")")}" }
//        require(a.shape[0] == b.shape[0]) { "The first dimensions of the ndarrays a and b must be match: ${a.shape[0]}(a.shape[0]) != ${b.shape[0]}(b.shape[0]" }
//        val ret = if (b.consistent) (b as NDArray).clone() else (b as NDArray).deepCopy()
//        val nhrs = if (ret.dim.d == 1) ret.shape[0] else ret.shape[1]
//
//        val info: Int = when (a.dtype) {
//            DataType.FloatDataType -> JniLinAlg.solve(
//                a.shape.first(), nhrs, a.data.getFloatArray().copyOf(), a.strides.first(),
//                ret.data.getFloatArray(), ret.strides.first()
//            )
//            DataType.DoubleDataType -> JniLinAlg.solve(
//                a.shape.first(), nhrs, a.data.getDoubleArray().copyOf(), a.strides.first(),
//                ret.data.getDoubleArray(), ret.strides.first()
//            )
//            else -> throw UnsupportedOperationException()
//        }
//        if (info > 0) {
//            throw Exception("The diagonal element of the triangular factor of a is zero, so that A is singular. The solution could not be computed.")
//        }
//
//        return ret
//    }

    //TODO (Double and Number type)
//    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
//        requireDotShape(a, b)
//        val transA = a.isTransposed()
//
//        return if (b.dim.d == 2) {
//            val shape = intArrayOf(a.shape[0], b.shape[1])
//            val size = shape.reduce(Int::times)
//            val transB = b.isTransposed()
//            val aN = if (transA || a.consistent) a else a.deepCopy()
//            val bN = if (transB || b.consistent) b else b.deepCopy()
//            when (a.dtype) {
//                DataType.FloatDataType -> {
//                    val c = FloatArray(size)
//                    JniLinAlg.dot(transA, aN.data.getFloatArray(), shape[0], shape[1], transB, bN.data.getFloatArray(), aN.shape[1], c)
//                    D2Array(MemoryViewFloatArray(c), 0, shape, dtype = DataType.FloatDataType, dim = D2)
//                }
//                DataType.DoubleDataType -> {
//                    val c = DoubleArray(size)
//                    JniLinAlg.dot(
//                        transA,
//                        aN.data.getDoubleArray(),
//                        shape[0],
//                        shape[1],
//                        transB,
//                        bN.data.getDoubleArray(),
//                        aN.shape[1],
//                        c
//                    )
//                    D2Array(MemoryViewDoubleArray(c), 0, shape, dtype = DataType.DoubleDataType, dim = D2)
//                }
//                else -> throw UnsupportedOperationException()
//            } as NDArray<T, D>
//        } else {
//            val shape = intArrayOf(a.shape[0])
//            val size = shape[0]
//            val aN = if (transA || a.consistent) a else a.deepCopy()
//            val bN = if (b.consistent) b else b.deepCopy()
//            when (a.dtype) {
//                DataType.FloatDataType -> {
//                    val c = FloatArray(size)
//                    JniLinAlg.dot(transA, aN.data.getFloatArray(), aN.shape[0], aN.shape[1], bN.data.getFloatArray(), c)
//                    D1Array(MemoryViewFloatArray(c), 0, shape, dtype = a.dtype, dim = D1)
//                }
//                DataType.DoubleDataType -> {
//                    val c = DoubleArray(size)
//                    JniLinAlg.dot(
//                        transA,
//                        aN.data.getDoubleArray(),
//                        aN.shape[0],
//                        aN.shape[1],
//                        bN.data.getDoubleArray(),
//                        c
//                    )
//                    D1Array(MemoryViewDoubleArray(c), 0, shape, dtype = a.dtype, dim = D1)
//                }
//                else -> throw UnsupportedOperationException()
//            } as NDArray<T, D>
//        }
//    }

//    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
//        require(a.size == b.size) { "Vector sizes don't match: ${a.size} != ${b.size}" }
//        return when (a.dtype) {
//            DataType.FloatDataType -> {
//                JniLinAlg.dot(
//                    a.size,
//                    a.data.getFloatArray(),
//                    a.strides.first(),
//                    b.data.getFloatArray(),
//                    b.strides.first()
//                )
//            }
//            DataType.DoubleDataType -> {
//                JniLinAlg.dot(
//                    a.size,
//                    a.data.getDoubleArray(),
//                    a.strides.first(),
//                    b.data.getDoubleArray(),
//                    b.strides.first()
//                )
//            }
//            else -> throw UnsupportedOperationException()
//        } as T
//    }
}