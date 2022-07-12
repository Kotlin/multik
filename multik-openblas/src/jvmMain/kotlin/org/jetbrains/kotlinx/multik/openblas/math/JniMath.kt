package org.jetbrains.kotlinx.multik.openblas.math

internal actual object JniMath {
    actual external fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int
    actual external fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int

    actual external fun exp(arr: FloatArray, size: Int): Boolean
    actual external fun exp(arr: DoubleArray, size: Int): Boolean
    actual external fun expC(arr: FloatArray, size: Int): Boolean
    actual external fun expC(arr: DoubleArray, size: Int): Boolean

    actual external fun log(arr: FloatArray, size: Int): Boolean
    actual external fun log(arr: DoubleArray, size: Int): Boolean
    actual external fun logC(arr: FloatArray, size: Int): Boolean
    actual external fun logC(arr: DoubleArray, size: Int): Boolean

    actual external fun sin(arr: FloatArray, size: Int): Boolean
    actual external fun sin(arr: DoubleArray, size: Int): Boolean
    actual external fun sinC(arr: FloatArray, size: Int): Boolean
    actual external fun sinC(arr: DoubleArray, size: Int): Boolean

    actual external fun cos(arr: FloatArray, size: Int): Boolean
    actual external fun cos(arr: DoubleArray, size: Int): Boolean
    actual external fun cosC(arr: FloatArray, size: Int): Boolean
    actual external fun cosC(arr: DoubleArray, size: Int): Boolean

    actual external fun <T : Number> array_max(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    actual external fun <T : Number> array_min(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    actual external fun <T : Number> sum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    actual external fun cumSum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: Any, axis: Int, dtype: Int): Boolean
}