package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

// =================================== Scalar + Array ===================================
// Returns a new array where this scalar is added to each element of [other].
// Always allocates via [deepCopy] (no mutation of the original).

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
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.plus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.plus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.plus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.plus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.plus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.plus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
}

/** Adds this scalar to each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.plus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = other.deepCopy() as NDArray
    ret += this
    return ret
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
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getByteArray()
    for (i in ret.indices) {
        data[i] = (this - data[i]).toByte()
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Short.minus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getShortArray()
    for (i in ret.indices) {
        data[i] = (this - data[i]).toShort()
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Int.minus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getIntArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Long.minus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getLongArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Float.minus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getFloatArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> Double.minus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getDoubleArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> ComplexFloat.minus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getComplexFloatArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
}

/** Subtracts each element of [other] from this scalar, returning a new array. */
public operator fun <D : Dimension> ComplexDouble.minus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getComplexDoubleArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return ret
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
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.times(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.times(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.times(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.times(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.times(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.times(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
}

/** Multiplies this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.times(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = other.deepCopy() as NDArray
    ret *= this
    return ret
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
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getByteArray()
    for (i in ret.indices) {
        data[i] = (this / data[i]).toByte()
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Short.div(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getShortArray()
    for (i in ret.indices) {
        data[i] = (this / data[i]).toShort()
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Int.div(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getIntArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Long.div(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getLongArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Float.div(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getFloatArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> Double.div(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getDoubleArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexFloat.div(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getComplexFloatArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}

/** Divides this scalar by each element of [other], returning a new array. */
public operator fun <D : Dimension> ComplexDouble.div(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = other.deepCopy() as NDArray
    val data = ret.data.getComplexDoubleArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return ret
}
