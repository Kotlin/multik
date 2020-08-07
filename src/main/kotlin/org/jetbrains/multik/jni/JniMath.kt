package org.jetbrains.multik.jni

import org.jetbrains.multik.api.Math
import org.jetbrains.multik.core.*

public object NativeMath : Math {
    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        return JniMath.argMax(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        return JniMath.argMin(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::exp)
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::log)
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::sin)
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::cos)
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        return JniMath.max(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        return JniMath.min(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        return JniMath.sum(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        val ret = D1Array<T>(initMemoryView(a.size, a.dtype), shape = intArrayOf(a.size), dtype = a.dtype, dim = D1)
        JniMath.cumSum(a.data.data, a.offset, a.size, a.shape, a.strides, ret.data.data, dtype = a.dtype.nativeCode)
        return ret
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): Ndarray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : Dimension> mathOperation(
        a: MultiArray<T, D>,
        function: (arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, out: DoubleArray, dtype: Int) -> Boolean
    ): Ndarray<Double, D> {
        val data = MemoryViewDoubleArray(DoubleArray(a.size))
        function(a.data.data, a.offset, a.size, a.shape, a.strides, data.data, a.dtype.nativeCode)
        return Ndarray<Double, D>(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
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