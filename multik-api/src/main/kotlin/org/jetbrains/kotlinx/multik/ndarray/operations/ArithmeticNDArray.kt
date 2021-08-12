/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Create a new array as the sum of [this] and [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

public operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Add [other] to [this]. Inplace operator.
 */
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
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
 * Add [other] element-wise. Inplace operator.
 */
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
 * Create a new array as difference between [this] and [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

public operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Subtract [other] from [this]. Inplace operator.
 */
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
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
 * Subtract [other] element-wise. Inplace operator.
 */
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
 * Create a new array as product of [this] and [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

public operator fun <T, D : Dimension> MultiArray<T, D>.times(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Multiply [this] by [other]. Inplace operator.
 */
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
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
 * Multiply [other] element-wise. Inplace operator.
 */
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
 * Create a new array as division of [this] by [other].
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

public operator fun <T, D : Dimension> MultiArray<T, D>.div(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Divide [this] by [other]. Inplace operator.
 */
public operator fun <T, D : Dimension> MutableMultiArray<T, D>.divAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
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
 * Divide by [other] element-wise. Inplace operator.
 */
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


private inline fun <T : Any, D : Dimension> MutableMultiArray<T, D>.commonAssignOp(
    other: Iterator<T>,
    op: (T, T) -> T
) {
    if (this.consistent) {
        for (i in this.indices)
            this.data[i] = op(this.data[i], other.next())
    } else {
        val left = this.asDNArray()
        for (index in this.multiIndices)
            left[index] = op(left[index], other.next())
    }
}

private inline fun <T : Any, D : Dimension> MutableMultiArray<T, D>.commonAssignOp(other: T, op: (T, T) -> T) {
    when (dim.d) {
        1 -> {
            this as MutableMultiArray<T, D1>
            for (i in this.indices)
                this[i] = op(this[i], other)
        }
        2 -> {
            this as MutableMultiArray<T, D2>
            for (i in this.multiIndices)
                this[i[0], i[1]] = op(this[i[0], i[1]], other)
        }
        3 -> {
            this as MutableMultiArray<T, D3>
            for (i in this.multiIndices)
                this[i[0], i[1], i[2]] = op(this[i[0], i[1], i[2]], other)
        }
        4 -> {
            this as MutableMultiArray<T, D4>
            for (i in this.multiIndices)
                this[i[0], i[1], i[2], i[3]] = op(this[i[0], i[1], i[2], i[3]], other)
        }
        else -> {
            val left = this.asDNArray()
            for (index in this.multiIndices)
                left[index] = op(left[index], other)
        }
    }
}
