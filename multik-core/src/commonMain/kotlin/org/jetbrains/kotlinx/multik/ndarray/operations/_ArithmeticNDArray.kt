package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryView
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.MutableMultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.forEach
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.requireEqualShape
import org.jetbrains.kotlinx.multik.ndarray.data.set

/**
 * Negates every element, returning a new array.
 *
 * ```
 * val a = mk.ndarray(mk[1, -2, 3])
 * val b = -a // [-1, 2, -3]
 * ```
 *
 * @return a new [NDArray] with each element negated.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MultiArray<T, D>.unaryMinus(): NDArray<T, D> =
    when (dtype) {
        DataType.DoubleDataType -> (this as NDArray<Double, D>).map { -it }
        DataType.FloatDataType -> (this as NDArray<Float, D>).map { -it }
        DataType.IntDataType -> (this as NDArray<Int, D>).map { -it }
        DataType.LongDataType -> (this as NDArray<Long, D>).map { -it }
        DataType.ComplexFloatDataType -> (this as NDArray<ComplexFloat, D>).map { -it }
        DataType.ComplexDoubleDataType -> (this as NDArray<ComplexDouble, D>).map { -it }
        DataType.ShortDataType -> (this as NDArray<Short, D>).map { -it }
        DataType.ByteDataType -> (this as NDArray<Byte, D>).map { -it }
    } as NDArray<T, D>

/**
 * Adds two arrays element-wise, returning a new array.
 *
 * Both arrays must have the same shape. The originals are not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * val b = mk.ndarray(mk[10, 20, 30])
 * val c = a + b // [11, 22, 33]
 * ```
 *
 * @param other the array to add (not modified).
 * @return a new [NDArray] with the element-wise sums.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 * @see [plusAssign] for the in-place variant.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Adds a scalar to each element, returning a new array.
 *
 * @param other the scalar to add.
 * @return a new [NDArray] with the scalar added to every element.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Adds [other] to this array element-wise in place.
 *
 * @param other the array to add.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: MultiArray<T, D>) {
    requireEqualShape(this.shape, other.shape)
    if (this.consistent && other.consistent) {
        this.data += (other.data as MemoryView)
    } else {
        when (dtype) {
            DataType.DoubleDataType -> (this as NDArray<Double, D>).commonAssignOp(other.iterator() as Iterator<Double>) { a, b -> a + b }
            DataType.FloatDataType -> (this as NDArray<Float, D>).commonAssignOp(other.iterator() as Iterator<Float>) { a, b -> a + b }
            DataType.IntDataType -> (this as NDArray<Int, D>).commonAssignOp(other.iterator() as Iterator<Int>) { a, b -> a + b }
            DataType.LongDataType -> (this as NDArray<Long, D>).commonAssignOp(other.iterator() as Iterator<Long>) { a, b -> a + b }
            DataType.ComplexFloatDataType -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other.iterator() as Iterator<ComplexFloat>) { a, b -> a + b }
            DataType.ComplexDoubleDataType -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other.iterator() as Iterator<ComplexDouble>) { a, b -> a + b }
            DataType.ShortDataType -> (this as NDArray<Short, D>).commonAssignOp(other.iterator() as Iterator<Short>) { a, b -> (a + b).toShort() }
            DataType.ByteDataType -> (this as NDArray<Byte, D>).commonAssignOp(other.iterator() as Iterator<Byte>) { a, b -> (a + b).toByte() }
        }
    }
}


/**
 * Adds a scalar [other] to every element of this array in place.
 *
 * @param other the scalar to add.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: T) {
    if (this.consistent) {
        this.data += other
    } else {
        when (other) {
            is Double -> (this as NDArray<Double, D>).commonAssignOp(other) { a, b -> a + b }
            is Float -> (this as NDArray<Float, D>).commonAssignOp(other) { a, b -> a + b }
            is Int -> (this as NDArray<Int, D>).commonAssignOp(other) { a, b -> a + b }
            is Long -> (this as NDArray<Long, D>).commonAssignOp(other) { a, b -> a + b }
            is ComplexFloat -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other) { a, b -> a + b }
            is ComplexDouble -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other) { a, b -> a + b }
            is Short -> (this as NDArray<Short, D>).commonAssignOp(other) { a, b -> (a + b).toShort() }
            is Byte -> (this as NDArray<Byte, D>).commonAssignOp(other) { a, b -> (a + b).toByte() }
        }
    }
}

/**
 * Subtracts [other] from this array element-wise, returning a new array.
 *
 * Both arrays must have the same shape. The originals are not modified.
 *
 * ```
 * val a = mk.ndarray(mk[10, 20, 30])
 * val b = mk.ndarray(mk[1, 2, 3])
 * val c = a - b // [9, 18, 27]
 * ```
 *
 * @param other the array to subtract (not modified).
 * @return a new [NDArray] with the element-wise differences.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 * @see [minusAssign] for the in-place variant.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Subtracts a scalar from each element, returning a new array.
 *
 * @param other the scalar to subtract.
 * @return a new [NDArray] with the scalar subtracted from every element.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Subtracts [other] from this array element-wise in place.
 *
 * @param other the array to subtract.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: MultiArray<T, D>) {
    requireEqualShape(this.shape, other.shape)
    if (this.consistent && other.consistent) {
        this.data -= (other.data as MemoryView)
    } else {
        when (dtype) {
            DataType.DoubleDataType -> (this as NDArray<Double, D>).commonAssignOp(other.iterator() as Iterator<Double>) { a, b -> a - b }
            DataType.FloatDataType -> (this as NDArray<Float, D>).commonAssignOp(other.iterator() as Iterator<Float>) { a, b -> a - b }
            DataType.IntDataType -> (this as NDArray<Int, D>).commonAssignOp(other.iterator() as Iterator<Int>) { a, b -> a - b }
            DataType.LongDataType -> (this as NDArray<Long, D>).commonAssignOp(other.iterator() as Iterator<Long>) { a, b -> a - b }
            DataType.ComplexFloatDataType -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other.iterator() as Iterator<ComplexFloat>) { a, b -> a - b }
            DataType.ComplexDoubleDataType -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other.iterator() as Iterator<ComplexDouble>) { a, b -> a - b }
            DataType.ShortDataType -> (this as NDArray<Short, D>).commonAssignOp(other.iterator() as Iterator<Short>) { a, b -> (a - b).toShort() }
            DataType.ByteDataType -> (this as NDArray<Byte, D>).commonAssignOp(other.iterator() as Iterator<Byte>) { a, b -> (a - b).toByte() }
        }
    }
}

/**
 * Subtracts a scalar [other] from every element of this array in place.
 *
 * @param other the scalar to subtract.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: T) {
    if (this.consistent) {
        this.data -= other
    } else {
        when (other) {
            is Double -> (this as NDArray<Double, D>).commonAssignOp(other) { a, b -> a - b }
            is Float -> (this as NDArray<Float, D>).commonAssignOp(other) { a, b -> a - b }
            is Int -> (this as NDArray<Int, D>).commonAssignOp(other) { a, b -> a - b }
            is Long -> (this as NDArray<Long, D>).commonAssignOp(other) { a, b -> a - b }
            is ComplexFloat -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other) { a, b -> a - b }
            is ComplexDouble -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other) { a, b -> a - b }
            is Short -> (this as NDArray<Short, D>).commonAssignOp(other) { a, b -> (a - b).toShort() }
            is Byte -> (this as NDArray<Byte, D>).commonAssignOp(other) { a, b -> (a - b).toByte() }
        }
    }
}

/**
 * Multiplies two arrays element-wise, returning a new array.
 *
 * This is element-wise multiplication, not matrix multiplication.
 * Both arrays must have the same shape. The originals are not modified.
 *
 * ```
 * val a = mk.ndarray(mk[2, 3, 4])
 * val b = mk.ndarray(mk[10, 20, 30])
 * val c = a * b // [20, 60, 120]
 * ```
 *
 * @param other the array to multiply with (not modified).
 * @return a new [NDArray] with the element-wise products.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 * @see [timesAssign] for the in-place variant.
 * @see [org.jetbrains.kotlinx.multik.api.linalg.LinAlg.dot] for matrix multiplication.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Multiplies each element by a scalar, returning a new array.
 *
 * @param other the scalar multiplier.
 * @return a new [NDArray] with every element multiplied by [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Multiplies this array by [other] element-wise in place.
 *
 * @param other the array to multiply with.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: MultiArray<T, D>) {
    requireEqualShape(this.shape, other.shape)
    if (this.consistent && other.consistent) {
        this.data *= (other.data as MemoryView)
    } else {
        when (dtype) {
            DataType.DoubleDataType -> (this as NDArray<Double, D>).commonAssignOp(other.iterator() as Iterator<Double>) { a, b -> a * b }
            DataType.FloatDataType -> (this as NDArray<Float, D>).commonAssignOp(other.iterator() as Iterator<Float>) { a, b -> a * b }
            DataType.IntDataType -> (this as NDArray<Int, D>).commonAssignOp(other.iterator() as Iterator<Int>) { a, b -> a * b }
            DataType.LongDataType -> (this as NDArray<Long, D>).commonAssignOp(other.iterator() as Iterator<Long>) { a, b -> a * b }
            DataType.ComplexFloatDataType -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other.iterator() as Iterator<ComplexFloat>) { a, b -> a * b }
            DataType.ComplexDoubleDataType -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other.iterator() as Iterator<ComplexDouble>) { a, b -> a * b }
            DataType.ShortDataType -> (this as NDArray<Short, D>).commonAssignOp(other.iterator() as Iterator<Short>) { a, b -> (a * b).toShort() }
            DataType.ByteDataType -> (this as NDArray<Byte, D>).commonAssignOp(other.iterator() as Iterator<Byte>) { a, b -> (a * b).toByte() }
        }
    }
}

/**
 * Multiplies every element of this array by a scalar [other] in place.
 *
 * @param other the scalar multiplier.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: T) {
    if (this.consistent) {
        this.data *= other
    } else {
        when (other) {
            is Double -> (this as NDArray<Double, D>).commonAssignOp(other) { a, b -> a * b }
            is Float -> (this as NDArray<Float, D>).commonAssignOp(other) { a, b -> a * b }
            is Int -> (this as NDArray<Int, D>).commonAssignOp(other) { a, b -> a * b }
            is Long -> (this as NDArray<Long, D>).commonAssignOp(other) { a, b -> a * b }
            is ComplexFloat -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other) { a, b -> a * b }
            is ComplexDouble -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other) { a, b -> a * b }
            is Short -> (this as NDArray<Short, D>).commonAssignOp(other) { a, b -> (a * b).toShort() }
            is Byte -> (this as NDArray<Byte, D>).commonAssignOp(other) { a, b -> (a * b).toByte() }
        }
    }
}

/**
 * Divides this array by [other] element-wise, returning a new array.
 *
 * Both arrays must have the same shape. The originals are not modified.
 * Integer division truncates toward zero.
 *
 * ```
 * val a = mk.ndarray(mk[10.0, 20.0, 30.0])
 * val b = mk.ndarray(mk[2.0, 4.0, 5.0])
 * val c = a / b // [5.0, 5.0, 6.0]
 * ```
 *
 * @param other the divisor array (not modified).
 * @return a new [NDArray] with the element-wise quotients.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 * @see [divAssign] for the in-place variant.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Divides each element by a scalar, returning a new array.
 *
 * @param other the scalar divisor.
 * @return a new [NDArray] with every element divided by [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Divides this array by [other] element-wise in place.
 *
 * @param other the divisor array.
 * @throws IllegalArgumentException if the shapes of [this] and [other] differ.
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.divAssign(other: MultiArray<T, D>) {
    requireEqualShape(this.shape, other.shape)
    if (this.consistent && other.consistent) {
        this.data /= (other.data as MemoryView)
    } else {
        when (dtype) {
            DataType.DoubleDataType -> (this as NDArray<Double, D>).commonAssignOp(other.iterator() as Iterator<Double>) { a, b -> a / b }
            DataType.FloatDataType -> (this as NDArray<Float, D>).commonAssignOp(other.iterator() as Iterator<Float>) { a, b -> a / b }
            DataType.IntDataType -> (this as NDArray<Int, D>).commonAssignOp(other.iterator() as Iterator<Int>) { a, b -> a / b }
            DataType.LongDataType -> (this as NDArray<Long, D>).commonAssignOp(other.iterator() as Iterator<Long>) { a, b -> a / b }
            DataType.ComplexFloatDataType -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other.iterator() as Iterator<ComplexFloat>) { a, b -> a / b }
            DataType.ComplexDoubleDataType -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other.iterator() as Iterator<ComplexDouble>) { a, b -> a / b }
            DataType.ShortDataType -> (this as NDArray<Short, D>).commonAssignOp(other.iterator() as Iterator<Short>) { a, b -> (a / b).toShort() }
            DataType.ByteDataType -> (this as NDArray<Byte, D>).commonAssignOp(other.iterator() as Iterator<Byte>) { a, b -> (a / b).toByte() }
        }
    }
}

/**
 * Divides every element of this array by a scalar [other] in place.
 *
 * @param other the scalar divisor.
 * @throws ArithmeticException if [other] is zero (for integer types).
 */
@Suppress("unchecked_cast")
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.divAssign(other: T) {
    if (this.consistent) {
        this.data /= other
    } else {
        when (other) {
            is Double -> (this as NDArray<Double, D>).commonAssignOp(other) { a, b -> a / b }
            is Float -> (this as NDArray<Float, D>).commonAssignOp(other) { a, b -> a / b }
            is Int -> (this as NDArray<Int, D>).commonAssignOp(other) { a, b -> a / b }
            is Long -> (this as NDArray<Long, D>).commonAssignOp(other) { a, b -> a / b }
            is ComplexFloat -> (this as NDArray<ComplexFloat, D>).commonAssignOp(other) { a, b -> a / b }
            is ComplexDouble -> (this as NDArray<ComplexDouble, D>).commonAssignOp(other) { a, b -> a / b }
            is Short -> (this as NDArray<Short, D>).commonAssignOp(other) { a, b -> (a / b).toShort() }
            is Byte -> (this as NDArray<Byte, D>).commonAssignOp(other) { a, b -> (a / b).toByte() }
        }
    }
}


/** Applies [op] pair-wise between this array's elements and [other], storing results in this array. */
internal inline fun <T : Any, D : Dimension> MutableMultiArray<T, D>.commonAssignOp(
    other: Iterator<T>, op: (T, T) -> T
) {
    if (this.consistent) {
        for (i in this.indices)
            this.data[i] = op(this.data[i], other.next())
    } else {
        this.multiIndices.forEach { index ->
            this[index] = op(this[index], other.next())
        }
    }
}

/** Applies [op] between each element of this array and the scalar [other], storing results in this array. */
@Suppress("unchecked_cast")
private inline fun <T : Any, D : Dimension> MutableMultiArray<T, D>.commonAssignOp(other: T, op: (T, T) -> T) {
    if (dim.d == 1) {
        this as MutableMultiArray<T, D1>
        for (i in this.indices)
            this[i] = op(this[i], other)
    } else {
        this.multiIndices.forEach { index ->
            this[index] = op(this[index], other)
        }
    }
}
