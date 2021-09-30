package org.jetbrains.kotlinx.multik.jni.math

internal object JniMath {
    external fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int
    external fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int

    external fun exp(arr: FloatArray, size: Int): Boolean
    external fun exp(arr: DoubleArray, size: Int): Boolean
    external fun expC(arr: FloatArray, size: Int): Boolean
    external fun expC(arr: DoubleArray, size: Int): Boolean

    external fun log(arr: FloatArray, size: Int): Boolean
    external fun log(arr: DoubleArray, size: Int): Boolean
    external fun logC(arr: FloatArray, size: Int): Boolean
    external fun logC(arr: DoubleArray, size: Int): Boolean

    external fun sin(arr: FloatArray, size: Int): Boolean
    external fun sin(arr: DoubleArray, size: Int): Boolean
    external fun sinC(arr: FloatArray, size: Int): Boolean
    external fun sinC(arr: DoubleArray, size: Int): Boolean

    external fun cos(arr: FloatArray, size: Int): Boolean
    external fun cos(arr: DoubleArray, size: Int): Boolean
    external fun cosC(arr: FloatArray, size: Int): Boolean
    external fun cosC(arr: DoubleArray, size: Int): Boolean

    external fun <T : Number> max(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    external fun <T : Number> min(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    external fun <T : Number> sum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    external fun cumSum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: Any, axis: Int = -1, dtype: Int): Boolean
}