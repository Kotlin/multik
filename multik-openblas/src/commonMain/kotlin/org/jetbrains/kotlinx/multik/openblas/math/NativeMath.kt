/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.math

import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.ndarray.data.*

internal object NativeMath : Math {

    override val mathEx: MathEx
        get() = NativeMathEx

    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        val strides: IntArray? = if (a.consistent) null else a.strides
        return JniMath.argMax(a.data.data, a.offset, a.size, a.shape, strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        val strides: IntArray? = if (a.consistent) null else a.strides
        return JniMath.argMin(a.data.data, a.offset, a.size, a.shape, strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        val strides: IntArray? = if (a.consistent) null else a.strides
        return when (a.dtype) {
            DataType.ByteDataType -> JniMath.array_max(a.data.getByteArray(), a.offset, a.size, a.shape, strides)
            DataType.ShortDataType -> JniMath.array_max(a.data.getShortArray(), a.offset, a.size, a.shape, strides)
            DataType.IntDataType -> JniMath.array_max(a.data.getIntArray(), a.offset, a.size, a.shape, strides)
            DataType.LongDataType -> JniMath.array_max(a.data.getLongArray(), a.offset, a.size, a.shape, strides)
            DataType.FloatDataType -> JniMath.array_max(a.data.getFloatArray(), a.offset, a.size, a.shape, strides)
            DataType.DoubleDataType -> JniMath.array_max(a.data.getDoubleArray(), a.offset, a.size, a.shape, strides)
            else -> TODO("ComplexFloat and ComplexDouble not yet implemented")
        } as T
    }

    override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        val strides: IntArray? = if (a.consistent) null else a.strides
        return when (a.dtype) {
            DataType.ByteDataType -> JniMath.array_min(a.data.getByteArray(), a.offset, a.size, a.shape, strides)
            DataType.ShortDataType -> JniMath.array_min(a.data.getShortArray(), a.offset, a.size, a.shape, strides)
            DataType.IntDataType -> JniMath.array_min(a.data.getIntArray(), a.offset, a.size, a.shape, strides)
            DataType.LongDataType -> JniMath.array_min(a.data.getLongArray(), a.offset, a.size, a.shape, strides)
            DataType.FloatDataType -> JniMath.array_min(a.data.getFloatArray(), a.offset, a.size, a.shape, strides)
            DataType.DoubleDataType -> JniMath.array_min(a.data.getDoubleArray(), a.offset, a.size, a.shape, strides)
            else -> TODO("ComplexFloat and ComplexDouble not yet implemented")
        } as T
    }

    override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        val strides: IntArray? = if (a.consistent) null else a.strides
        return when (a.dtype) {
            DataType.ByteDataType -> JniMath.sum(a.data.getByteArray(), a.offset, a.size, a.shape, strides)
            DataType.ShortDataType -> JniMath.sum(a.data.getShortArray(), a.offset, a.size, a.shape, strides)
            DataType.IntDataType -> JniMath.sum(a.data.getIntArray(), a.offset, a.size, a.shape, strides)
            DataType.LongDataType -> JniMath.sum(a.data.getLongArray(), a.offset, a.size, a.shape, strides)
            DataType.FloatDataType -> JniMath.sum(a.data.getFloatArray(), a.offset, a.size, a.shape, strides)
            DataType.DoubleDataType -> JniMath.sum(a.data.getDoubleArray(), a.offset, a.size, a.shape, strides)
            else -> TODO("ComplexFloat and ComplexDouble not yet implemented")
        } as T
    }

    override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        val ret = D1Array<T>(initMemoryView(a.size, a.dtype), shape = intArrayOf(a.size), dim = D1)
        val strides: IntArray? = if (a.consistent) null else a.strides
        when (a.dtype) {
            DataType.ByteDataType -> JniMath.cumSum(a.data.getByteArray(), a.offset, a.size, a.shape, strides, ret.data.getByteArray())
            DataType.ShortDataType -> JniMath.cumSum(a.data.getShortArray(), a.offset, a.size, a.shape, strides, ret.data.getShortArray())
            DataType.IntDataType -> JniMath.cumSum(a.data.getIntArray(), a.offset, a.size, a.shape, strides, ret.data.getIntArray())
            DataType.LongDataType -> JniMath.cumSum(a.data.getLongArray(), a.offset, a.size, a.shape, strides, ret.data.getLongArray())
            DataType.FloatDataType -> JniMath.cumSum(a.data.getFloatArray(), a.offset, a.size, a.shape, strides, ret.data.getFloatArray())
            DataType.DoubleDataType -> JniMath.cumSum(a.data.getDoubleArray(), a.offset, a.size, a.shape, strides, ret.data.getDoubleArray())
            else -> TODO("ComplexFloat and ComplexDouble not yet implemented")
        }
        return ret
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> {
        TODO("Not yet implemented")
    }
}
