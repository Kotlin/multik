package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewByteArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewIntArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewLongArray
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewShortArray
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

// =================================== Scalar + Array ===================================
// Returns a new array where this scalar is added to each element of [other].
// Allocates a new contiguous result in a single pass (no mutation of the original).

/**
 * Adds this scalar to each element of [other], returning a new array.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * val b = 10.toByte() + a // [11, 12, 13]
 * ```
 *
 * @param other the source array (not modified).
 * @return a new [NDArray] with the sum.
 */
public operator fun <D : Dimension> Byte.plus(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val scalar = this
    return other.mapByte { (it + scalar).toByte() }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.plus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val scalar = this
    return other.mapShort { (it + scalar).toShort() }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.plus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val scalar = this
    return other.mapInt { it + scalar }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.plus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val scalar = this
    return other.mapLong { it + scalar }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.plus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val scalar = this
    return other.mapFloat { it + scalar }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.plus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val scalar = this
    return other.mapDouble { it + scalar }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.plus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val scalar = this
    return other.mapComplexFloat { it + scalar }
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.plus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val scalar = this
    return other.mapComplexDouble { it + scalar }
}

// =================================== Scalar - Array ===================================
// Returns a new array where each element of [other] is subtracted from this scalar.
// Note: the result is `scalar - element`, not `element - scalar`.

/**
 * Subtracts each element of [other] from this scalar, returning a new array.
 *
 * The result contains `this - other[i]` for each element. The original array is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * val b = 10 - a // [9, 8, 7]
 * ```
 *
 * @param other the source array (not modified).
 * @return a new [NDArray] with the differences.
 */
public operator fun <D : Dimension> Byte.minus(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val scalar = this
    return other.mapByte { (scalar - it).toByte() }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Short.minus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val scalar = this
    return other.mapShort { (scalar - it).toShort() }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Int.minus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val scalar = this
    return other.mapInt { scalar - it }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Long.minus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val scalar = this
    return other.mapLong { scalar - it }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Float.minus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val scalar = this
    return other.mapFloat { scalar - it }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Double.minus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val scalar = this
    return other.mapDouble { scalar - it }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> ComplexFloat.minus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val scalar = this
    return other.mapComplexFloat { scalar - it }
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> ComplexDouble.minus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val scalar = this
    return other.mapComplexDouble { scalar - it }
}

// =================================== Scalar * Array ===================================
// Returns a new array where this scalar is multiplied by each element of [other].
// This is element-wise multiplication, not matrix multiplication.

/**
 * Multiplies this scalar by each element of [other], returning a new array.
 *
 * This is element-wise multiplication. For matrix multiplication, use [mk.linalg.dot][org.jetbrains.kotlinx.multik.api.linalg.LinAlg.dot].
 * The original array is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * val b = 10.toByte() * a // [10, 20, 30]
 * ```
 *
 * @param other the source array (not modified).
 * @return a new [NDArray] with the products.
 */
public operator fun <D : Dimension> Byte.times(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val scalar = this
    return other.mapByte { (it * scalar).toByte() }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.times(other: MultiArray<Short, D>): NDArray<Short, D> {
    val scalar = this
    return other.mapShort { (it * scalar).toShort() }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.times(other: MultiArray<Int, D>): NDArray<Int, D> {
    val scalar = this
    return other.mapInt { it * scalar }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.times(other: MultiArray<Long, D>): NDArray<Long, D> {
    val scalar = this
    return other.mapLong { it * scalar }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.times(other: MultiArray<Float, D>): NDArray<Float, D> {
    val scalar = this
    return other.mapFloat { it * scalar }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.times(other: MultiArray<Double, D>): NDArray<Double, D> {
    val scalar = this
    return other.mapDouble { it * scalar }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.times(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val scalar = this
    return other.mapComplexFloat { it * scalar }
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.times(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val scalar = this
    return other.mapComplexDouble { it * scalar }
}

// =================================== Scalar / Array ===================================
// Returns a new array where this scalar is divided by each element of [other].
// Note: the result is `scalar / element`, not `element / scalar`.
// Integer division truncates toward zero.

/**
 * Divides this scalar by each element of [other], returning a new array.
 *
 * The result contains `this / other[i]` for each element. Integer division truncates
 * toward zero. The original array is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 5])
 * val b = 10.toByte() / a // [10, 5, 2]
 * ```
 *
 * @param other the source array (not modified).
 * @return a new [NDArray] with the quotients.
 * @throws ArithmeticException if any element of [other] is zero (for integer types).
 */
public operator fun <D : Dimension> Byte.div(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val scalar = this
    return other.mapByte { (scalar / it).toByte() }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.div(other: MultiArray<Short, D>): NDArray<Short, D> {
    val scalar = this
    return other.mapShort { (scalar / it).toShort() }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.div(other: MultiArray<Int, D>): NDArray<Int, D> {
    val scalar = this
    return other.mapInt { scalar / it }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.div(other: MultiArray<Long, D>): NDArray<Long, D> {
    val scalar = this
    return other.mapLong { scalar / it }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.div(other: MultiArray<Float, D>): NDArray<Float, D> {
    val scalar = this
    return other.mapFloat { scalar / it }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.div(other: MultiArray<Double, D>): NDArray<Double, D> {
    val scalar = this
    return other.mapDouble { scalar / it }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.div(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val scalar = this
    return other.mapComplexFloat { scalar / it }
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.div(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val scalar = this
    return other.mapComplexDouble { scalar / it }
}

// =================================== Internal kernels ===================================
// Single-pass scalar-on-left kernels. Each allocates the result buffer once and fills it in one
// traversal, reading the source's backing array directly: sequentially when the source is
// `consistent`, otherwise through a row-major odometer over [MultiArray.offset]/[MultiArray.strides].
//
// They are private and written out per dtype on purpose: the loops stay monomorphic and primitive
// on every backend, with no boxing on either path. Neither the public `map` nor [NDArray.deepCopy]
// can be reused here: both go through `Iterator<T>`, which boxes every element.

/**
 * Advances the row-major odometer [index] by one element and returns the flat position of the next
 * element, given the current position [p]. Amortized O(1): only the axes that roll over are touched.
 */
@Suppress("NOTHING_TO_INLINE")
private inline fun advanceIndex(index: IntArray, shape: IntArray, strides: IntArray, p: Int): Int {
    var position = p
    var axis = index.size - 1
    while (axis >= 0) {
        val next = index[axis] + 1
        if (next < shape[axis]) {
            index[axis] = next
            return position + strides[axis]
        }
        position -= strides[axis] * index[axis]
        index[axis] = 0
        axis--
    }
    return position
}

private inline fun <D : Dimension> MultiArray<Byte, D>.mapByte(transform: (Byte) -> Byte): NDArray<Byte, D> {
    val n = size
    val src = data.getByteArray()
    val out = ByteArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewByteArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<Short, D>.mapShort(transform: (Short) -> Short): NDArray<Short, D> {
    val n = size
    val src = data.getShortArray()
    val out = ShortArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewShortArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<Int, D>.mapInt(transform: (Int) -> Int): NDArray<Int, D> {
    val n = size
    val src = data.getIntArray()
    val out = IntArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewIntArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<Long, D>.mapLong(transform: (Long) -> Long): NDArray<Long, D> {
    val n = size
    val src = data.getLongArray()
    val out = LongArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewLongArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<Float, D>.mapFloat(transform: (Float) -> Float): NDArray<Float, D> {
    val n = size
    val src = data.getFloatArray()
    val out = FloatArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewFloatArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<Double, D>.mapDouble(transform: (Double) -> Double): NDArray<Double, D> {
    val n = size
    val src = data.getDoubleArray()
    val out = DoubleArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewDoubleArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<ComplexFloat, D>.mapComplexFloat(
    transform: (ComplexFloat) -> ComplexFloat
): NDArray<ComplexFloat, D> {
    val n = size
    val src = data.getComplexFloatArray()
    val out = ComplexFloatArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewComplexFloatArray(out), 0, shape.copyOf(), dim = dim)
}

private inline fun <D : Dimension> MultiArray<ComplexDouble, D>.mapComplexDouble(
    transform: (ComplexDouble) -> ComplexDouble
): NDArray<ComplexDouble, D> {
    val n = size
    val src = data.getComplexDoubleArray()
    val out = ComplexDoubleArray(n)
    if (consistent) {
        for (i in 0 until n) out[i] = transform(src[i])
    } else {
        val srcShape = shape
        val srcStrides = strides
        val index = IntArray(srcShape.size)
        var p = offset
        for (i in 0 until n) {
            out[i] = transform(src[p])
            p = advanceIndex(index, srcShape, srcStrides, p)
        }
    }
    return NDArray(MemoryViewComplexDoubleArray(out), 0, shape.copyOf(), dim = dim)
}
