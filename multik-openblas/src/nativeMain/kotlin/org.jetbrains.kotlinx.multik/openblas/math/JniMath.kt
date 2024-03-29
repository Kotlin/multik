package org.jetbrains.kotlinx.multik.openblas.math

import kotlinx.cinterop.*
import org.jetbrains.kotlinx.multik.cinterop.*

@OptIn(ExperimentalForeignApi::class)
internal actual object JniMath {
    actual fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int = when(arr) {
        is DoubleArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        is FloatArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        is IntArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        is LongArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        is ByteArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        is ShortArray -> arr.usePinned { argmin(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
        else -> throw Exception("Only primitive arrays are supported for Kotlin/Native `argMin`")
    }

    actual fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int = when(arr) {
    is DoubleArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    is FloatArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    is IntArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    is LongArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    is ByteArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    is ShortArray -> arr.usePinned { argmax(it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), dtype) }
    else -> throw Exception("Only primitive arrays are supported for Kotlin/Native `argMax`")
}

    actual fun exp(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_exp_float(it.addressOf(0), size)
        true // TODO(return Boolean)
    }
    actual fun exp(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_exp_double(it.addressOf(0), size)
        true
    }
    actual fun expC(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_exp_complex_float(it.addressOf(0), size)
        true
    }
    actual fun expC(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_exp_complex_double(it.addressOf(0), size)
        true
    }

    actual fun log(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_log_float(it.addressOf(0), size)
        true
    }
    actual fun log(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_log_double(it.addressOf(0), size)
        true
    }
    actual fun logC(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_log_complex_float(it.addressOf(0), size)
        true
    }
    actual fun logC(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_log_complex_double(it.addressOf(0), size)
        true
    }

    actual fun sin(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_sin_float(it.addressOf(0), size)
        true
    }
    actual fun sin(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_sin_double(it.addressOf(0), size)
        true
    }
    actual fun sinC(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_sin_complex_float(it.addressOf(0), size)
        true
    }
    actual fun sinC(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_sin_complex_double(it.addressOf(0), size)
        true
    }

    actual fun cos(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_cos_float(it.addressOf(0), size)
        true
    }
    actual fun cos(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_cos_double(it.addressOf(0), size)
        true
    }
    actual fun cosC(arr: FloatArray, size: Int): Boolean = arr.usePinned {
        array_cos_complex_float(it.addressOf(0), size)
        true
    }
    actual fun cosC(arr: DoubleArray, size: Int): Boolean = arr.usePinned {
        array_cos_complex_double(it.addressOf(0), size)
        true
    }

    actual fun array_max(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte =
        array_max_int8(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_max(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short =
        array_max_int16(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_max(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int =
        array_max_int32(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_max(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long =
        array_max_int64(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_max(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float =
        array_max_float(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_max(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double =
        array_max_double(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())

    actual fun array_min(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte =
        array_min_int8(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_min(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short =
        array_min_int16(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_min(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int =
        array_min_int32(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_min(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long =
        array_min_int64(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_min(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float =
        array_min_float(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun array_min(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double =
        array_min_double(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())

    actual fun sum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte =
        array_sum_int8(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun sum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short =
        array_sum_int16(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun sum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int =
        array_sum_int32(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun sum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long =
        array_sum_int64(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun sum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float =
        array_sum_float(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())
    actual fun sum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double =
        array_sum_double(arr.toCValues(), offset, size, shape.size, shape.toCValues(), strides?.toCValues())

    actual fun cumSum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ByteArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 1)
        }
        arrPin.unpin()
        return true
    }
    actual fun cumSum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ShortArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 2)
        }
        arrPin.unpin()
        return true
    }

    actual fun cumSum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: IntArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 3)
        }
        arrPin.unpin()
        return true
    }

    actual fun cumSum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: LongArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 4)
        }
        arrPin.unpin()
        return true
    }

    actual fun cumSum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: FloatArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 5)
        }
        arrPin.unpin()
        return true
    }

    actual fun cumSum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: DoubleArray, axis: Int): Boolean {
        val arrPin = arr.pin()
        out.usePinned {
            array_cumsum(arrPin.addressOf(0), it.addressOf(0), offset, size, shape.size, shape.toCValues(), strides?.toCValues(), 6)
        }
        arrPin.unpin()
        return true
    }
}