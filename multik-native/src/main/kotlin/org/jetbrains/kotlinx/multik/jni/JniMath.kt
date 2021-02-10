/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.api.Math
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object NativeMath : Math {
    init {
        NativeEngine
    }

    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        return JniMath.argMax(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
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

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        return JniMath.argMin(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
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

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a, JniMath::exp)
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a, JniMath::log)
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a, JniMath::sin)
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a, JniMath::cos)
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        return JniMath.max(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
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

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        return JniMath.min(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
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

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        return JniMath.sum(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
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

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        val ret = D1Array<T>(initMemoryView(a.size, a.dtype), shape = intArrayOf(a.size), dtype = a.dtype, dim = D1)
        JniMath.cumSum(a.data.data, a.offset, a.size, a.shape, a.strides, ret.data.data, dtype = a.dtype.nativeCode)
        return ret
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : Dimension> mathOperation(
        a: MultiArray<T, D>,
        function: (arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, out: DoubleArray, dtype: Int) -> Boolean
    ): NDArray<Double, D> {
        val data = MemoryViewDoubleArray(DoubleArray(a.size))
        function(a.data.data, a.offset, a.size, a.shape, a.strides, data.data, a.dtype.nativeCode)
        return NDArray<Double, D>(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }

}

private object JniMath {
    external fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, dtype: Int): Int
    external fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, dtype: Int): Int
    external fun exp(
        arr: Any, offset: Int, size: Int, shape: IntArray,
        strides: IntArray, out: DoubleArray, dtype: Int
    ): Boolean

    external fun log(
        arr: Any, offset: Int, size: Int, shape: IntArray,
        strides: IntArray, out: DoubleArray, dtype: Int
    ): Boolean

    external fun sin(
        arr: Any, offset: Int, size: Int, shape: IntArray,
        strides: IntArray, out: DoubleArray, dtype: Int
    ): Boolean

    external fun cos(
        arr: Any, offset: Int, size: Int, shape: IntArray,
        strides: IntArray, out: DoubleArray, dtype: Int
    ): Boolean

    external fun <T : Number> max(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, dtype: Int): T
    external fun <T : Number> min(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, dtype: Int): T
    external fun <T : Number> sum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, dtype: Int): T
    external fun cumSum(
        arr: Any, offset: Int, size: Int, shape: IntArray,
        strides: IntArray, out: Any, axis: Int = -1, dtype: Int
    ): Boolean
}