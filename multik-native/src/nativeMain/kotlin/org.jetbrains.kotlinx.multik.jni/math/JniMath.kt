package org.jetbrains.kotlinx.multik.jni.math

import kotlinx.cinterop.StableRef
import kotlinx.cinterop.toCValues
import org.jetbrains.kotlinx.multik.cinterop.*

internal actual object JniMath {
    actual fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int =
        argmin(StableRef.create(arr).asCPointer(), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype)
    actual fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int =
        argmax(StableRef.create(arr).asCPointer(), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype)

    actual fun exp(arr: FloatArray, size: Int): Boolean {
        array_exp_float(arr.toCValues(), size)
        return true
    }
    actual fun exp(arr: DoubleArray, size: Int): Boolean {
        array_exp_double(arr.toCValues(), size)
        return true
    }
    actual fun expC(arr: FloatArray, size: Int): Boolean {
        array_exp_complex_float(arr.toCValues(), size)
        return true
    }
    actual fun expC(arr: DoubleArray, size: Int): Boolean {
        array_exp_complex_double(arr.toCValues(), size)
        return true
    }


    actual fun log(arr: FloatArray, size: Int): Boolean {
        array_log_float(arr.toCValues(), size)
        return true
    }
    actual fun log(arr: DoubleArray, size: Int): Boolean {
        array_log_double(arr.toCValues(), size)
        return true
    }
    actual fun logC(arr: FloatArray, size: Int): Boolean {
        array_log_complex_float(arr.toCValues(), size)
        return true
    }
    actual fun logC(arr: DoubleArray, size: Int): Boolean {
        array_log_complex_double(arr.toCValues(), size)
        return true
    }

    actual fun sin(arr: FloatArray, size: Int): Boolean {
        array_log_float(arr.toCValues(), size)
        return true
    }
    actual fun sin(arr: DoubleArray, size: Int): Boolean {
        array_log_double(arr.toCValues(), size)
        return true
    }
    actual fun sinC(arr: FloatArray, size: Int): Boolean {
        array_log_complex_float(arr.toCValues(), size)
        return true
    }
    actual fun sinC(arr: DoubleArray, size: Int): Boolean {
        array_log_complex_double(arr.toCValues(), size)
        return true
    }

    actual fun cos(arr: FloatArray, size: Int): Boolean {
        array_cos_float(arr.toCValues(), size)
        return true
    }
    actual fun cos(arr: DoubleArray, size: Int): Boolean {
        array_cos_double(arr.toCValues(), size)
        return true
    }
    actual fun cosC(arr: FloatArray, size: Int): Boolean {
        array_cos_complex_float(arr.toCValues(), size)
        return true
    }
    actual fun cosC(arr: DoubleArray, size: Int): Boolean {
        array_cos_complex_double(arr.toCValues(), size)
        return true
    }

    actual fun <T : Number> array_max(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T {
        return when (dtype) {
            1 -> array_max_int8((arr as ByteArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            2 -> array_max_int16((arr as ShortArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            3 -> array_max_int32((arr as IntArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            4 -> array_max_int64((arr as LongArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            5 -> array_max_float((arr as FloatArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            6 -> array_max_double((arr as DoubleArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            else -> throw Exception()
        } as T
    }
    actual fun <T : Number> array_min(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T {
        return when (dtype) {
            1 -> array_min_int8((arr as ByteArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            2 -> array_min_int16((arr as ShortArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            3 -> array_min_int32((arr as IntArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            4 -> array_min_int64((arr as LongArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            5 -> array_min_float((arr as FloatArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            6 -> array_min_double((arr as DoubleArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            else -> throw Exception()
        } as T
    }
    actual fun <T : Number> sum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T {
        return when (dtype) {
            1 -> array_sum_int8((arr as ByteArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            2 -> array_sum_int16((arr as ShortArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            3 -> array_sum_int32((arr as IntArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            4 -> array_sum_int64((arr as LongArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            5 -> array_sum_float((arr as FloatArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            6 -> array_sum_double((arr as DoubleArray).toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
            else -> throw Exception()
        } as T
    }

    actual fun cumSum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: Any, axis: Int, dtype: Int): Boolean {
        array_cumsum(StableRef.create(arr).asCPointer(), StableRef.create(out).asCPointer(), offset, size,
            shape.size, shape.toCValues(), strides?.toCValues(), dtype)
        return true
    }
}