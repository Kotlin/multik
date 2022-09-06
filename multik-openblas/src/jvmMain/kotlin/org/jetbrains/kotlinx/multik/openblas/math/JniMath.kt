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

    actual external fun array_max(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    actual external fun array_max(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    actual external fun array_max(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    actual external fun array_max(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    actual external fun array_max(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    actual external fun array_max(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    actual external fun array_min(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    actual external fun array_min(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    actual external fun array_min(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    actual external fun array_min(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    actual external fun array_min(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    actual external fun array_min(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    actual external fun sum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    actual external fun sum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    actual external fun sum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    actual external fun sum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    actual external fun sum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    actual external fun sum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    actual external fun cumSum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ByteArray, axis: Int): Boolean
    actual external fun cumSum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ShortArray, axis: Int): Boolean
    actual external fun cumSum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: IntArray, axis: Int): Boolean
    actual external fun cumSum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: LongArray, axis: Int): Boolean
    actual external fun cumSum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: FloatArray, axis: Int): Boolean
    actual external fun cumSum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: DoubleArray, axis: Int): Boolean
}