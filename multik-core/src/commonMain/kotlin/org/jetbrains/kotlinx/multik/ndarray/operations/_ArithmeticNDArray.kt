/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Returns a new NDArray object with all elements negated for the given MultiArray object.
 *
 * @return The NDArray object with all elements negated.
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
 * Calculates the sum of [this] MultiArray and [other] MultiArray, resulting in a new NDArray with the same
 * Dimension type as the input arrays.
 *
 * @param other MultiArray to be added to [this] MultiArray
 * @return NDArray<T, D>, the sum of [this] MultiArray and [other] MultiArray
 * @throws IllegalArgumentException if the shape of [this] and [other] [MultiArray] are not equal.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Returns a new NDArray with the elements of this MultiArray added with the given element value
 *
 * @param other the element to be added. Must be of the same type as the elements in MultiArray
 * @return a new NDArray object with the same dimensions as this MultiArray but with passed element added to each element
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Adds the elements of [other] to [this] MultiArray instance and returns this instance.
 * This is an in-place operation.
 *
 * @param other the MultiArray instance to add
 * @throws IllegalArgumentException if the shapes of both arrays are not equal
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
 * Adds an element [other] element-wise to the current [MutableMultiArray], modifying it in-place.
 *
 * @param other The element to be added to the [MutableMultiArray].
 * @throws ClassCastException If [other] is not one of the following types: Double, Float,
 * Int, Long, ComplexFloat, ComplexDouble, Short, or Byte.
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
 * Calculates the difference of [this] MultiArray and [other] MultiArray, resulting in a new NDArray with the same
 * Dimension type as the input arrays.
 *
 * @param other MultiArray to be subtracted from [this] [MultiArray].
 * @return Returns a new [NDArray] object which is the difference between [this] and [other] [MultiArray].
 * @throws IllegalArgumentException if the shape of [this] and [other] [MultiArray] are not equal.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Returns a new NDArray resulting from the subtraction of the given value from all the elements of the MultiArray.
 *
 * @param other the value to subtract from the elements of the MultiArray
 * @return a new NDArray representing the result of the subtraction
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Subtract [other] from [this] element-wise in place.
 *
 *  * If both the arrays have the same shape, this method performs an in-place subtract operation and assigns the
 *  * result to this [MutableMultiArray].
 *  * Otherwise, it performs element-wise subtraction of this array and [other] array,
 *  * and assigns the result to this [MutableMultiArray].
 *
 * @param other The array to subtract from this.
 * @throws IllegalArgumentException If the shapes of [this] and [other] are not equal.
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
 * Subtract [other] element-wise from the current array. This is an inplace operator.
 *
 * @param other The element to subtract from the current array.
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
 * Multiplies this [MultiArray] with [other] [MultiArray] element-wise to produce a new [NDArray].
 *
 * @param other the [MultiArray] to be multiplied with [this]
 * @return an [NDArray] formed by the element-wise multiplication of [this] and [other] [MultiArray]
 * @throws IllegalArgumentException in case the shapes of [this] and [other] [MultiArray] do not match.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Performs multiplication operation between MultiArray and scalar value.
 * @param other The scalar value of type T to be multiplied.
 * @return NDArray object of type T and dimension D after performing multiplication operation.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Multiplies this [MutableMultiArray] by the [other] [MultiArray] element-wise in place.
 *
 * If both the arrays have the same shape, this method performs an in-place multiplication operation and assigns the
 * result to this [MutableMultiArray].
 * Otherwise, it performs element-wise multiplication of this array and [other] array,
 * and assigns the result to this [MutableMultiArray].
 *
 * @param other the [MultiArray] to be multiplied element-wise with this [MutableMultiArray]
 * @throws IllegalArgumentException if both arrays do not have the same shape.
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
 * Multiplies the [other] element-wise with the current [MutableMultiArray] and updates the current array in place.
 *
 * @param other the value to be multiplied element-wise with [MutableMultiArray]
 * @throws ClassCastException if [other] is not a compatible data type for the operation
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
 * Creates a new NDArray as a division of [this] MultiArray object by [other] MultiArray.
 *
 * @param other The MultiArray object to be divided by.
 * @return A new NDArray object containing the result of the division operation.
 * @throws IllegalArgumentException if the shape of [this] and [other] are not equal.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): NDArray<T, D> {
    requireEqualShape(this.shape, other.shape)
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Returns a new NDArray<T, D> resulting from each element of the MultiArray<T, D>
 * being divided by the specified `other` element.
 *
 * @param other The element to divide each element of the MultiArray by.
 * @return A new NDArray<T, D> with the result of the division operation.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).copy() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Divide this [MutableMultiArray] by another [MultiArray] element-wise.
 * This method performs the division operation in place, and modifies the original array.
 *
 * If both the arrays have the same shape, this method performs an in-place division operation and assigns the
 * result to this [MutableMultiArray].
 * Otherwise, it performs element-wise division of this array and [other] array,
 * and assigns the result to this [MutableMultiArray].
 *
 * @param other the MultiArray to divide by
 * @throws IllegalArgumentException if [this] and [other] have different shapes
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
 * Divide each element of the multi-dimensional array in place by another element [other].
 *
 * @param other The element to divide by.
 * @throws ArithmeticException if [other] is zero or causes an overflow during the division
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


/**
 * Performs a common assignment operation on a MutableMultiArray.
 *
 * @param other An iterator of data is the same type as the MutableMultiArray.
 * @param op A lambda function that takes two arguments of type T and returns a value of the same type.
 * This function is used to perform the common assignment operation.
 * @throws NoSuchElementException If the iterator passed as `other` does not contain enough elements.
 */
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

/**
 * Applies the given operator `op` to each element of `this` and `other` and
 * stores the result in `this`.
 *
 * @param other The element to apply the operator to.
 * @param op The operator function to apply.
 * @throws IndexOutOfBoundsException If `this` and `other` are not the same size.
 */
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
