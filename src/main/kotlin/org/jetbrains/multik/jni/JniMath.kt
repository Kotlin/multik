package org.jetbrains.multik.jni

import org.jetbrains.multik.api.Math
import org.jetbrains.multik.core.*
import org.jetbrains.multik.core.MemoryViewByteArray

object NativeMath : Math {
    override fun <T : Number, D : DN> argMax(a: Ndarray<T, D>): Int {
        return JniMath.argMax(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : DN> argMin(a: Ndarray<T, D>): Int {
        return JniMath.argMin(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : DN> exp(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::exp)
//        val data = MemoryViewDoubleArray(DoubleArray(a.size))
//        JniMath.exp(a.data.data, a.offset, a.size, a.shape, a.strides, data.data, a.dtype.nativeCode)
//        return initNdarray(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }

    override fun <T : Number, D : DN> log(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::log)
//        val data = MemoryViewDoubleArray(DoubleArray(a.size))
//        JniMath.log(a.data.data, a.offset, a.size, a.shape, a.strides, data.data, a.dtype.nativeCode)
//        return initNdarray(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }

    override fun <T : Number, D : DN> sin(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::sin)
    }

    override fun <T : Number, D : DN> cos(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a, JniMath::cos)
    }

    override fun <T : Number, D : DN> max(a: Ndarray<T, D>): T {
        return JniMath.max(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : DN> min(a: Ndarray<T, D>): T {
        return JniMath.min(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : DN> sum(a: Ndarray<T, D>): T {
        return JniMath.sum(a.data.data, a.offset, a.size, a.shape, a.strides, a.dtype.nativeCode)
    }

    override fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>): D1Array<T> {
        val ret = D1Array<T>(initMemoryView(a.size, a.dtype), shape = intArrayOf(a.size), dtype = a.dtype)
        JniMath.cumSum(a.data.data, a.offset, a.size, a.shape, a.strides, ret.data.data, dtype = a.dtype.nativeCode)
        return ret
    }

    override fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>, axis: Int): Ndarray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : DN> mathOperation(
        a: Ndarray<T, D>,
        function: (arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray, out: DoubleArray, dtype: Int) -> Boolean
    ): Ndarray<Double, D> {
        val data = MemoryViewDoubleArray(DoubleArray(a.size))
        function(a.data.data, a.offset, a.size, a.shape, a.strides, data.data, a.dtype.nativeCode)
        return initNdarray(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
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