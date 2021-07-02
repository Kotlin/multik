/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Create a new array as the sum of [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.plus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret += other
    return ret
}

/**
 * Add [other] to [this]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
    when {
        this.consistent && other.consistent -> {
            this.data += (other.data as MemoryView)
        }
        this.consistent && !other.consistent -> {
            val iterRight = other.iterator()
            for (i in this.indices)
                this.data[i] += iterRight.next()
        }
        else -> {
            val left = this.asDNArray()
            val iterRight = other.iterator()
            for (index in this.multiIndices)
                left[index] += iterRight.next()
        }
    }
}


/**
 * Add [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: T) {
    when {
        this.consistent -> {
            this.data += other
        }
        dim.d == 1 -> {
            this as MutableMultiArray<T, D1>
            for (i in this.indices)
                this[i] += other
        }
        else -> {
            val left = this.asDNArray()
            for (index in this.multiIndices)
                left[index] += other
        }
    }
}

/**
 * Create a new array as difference between [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.minus(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret -= other
    return ret
}

/**
 * Subtract [other] from [this]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
    when {
        this.consistent && other.consistent -> {
            this.data -= (other.data as MemoryView)
        }
        this.consistent && !other.consistent -> {
            val iterRight = other.iterator()
            for (i in this.indices)
                this.data[i] -= iterRight.next()
        }
        else -> {
            val left = this.asDNArray()
            val iterRight = other.iterator()
            for (index in this.multiIndices)
                left[index] -= iterRight.next()
        }
    }
}

/**
 * Subtract [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: T) {
    when {
        this.consistent -> {
            this.data -= other
        }
        dim.d == 1 -> {
            this as MutableMultiArray<T, D1>
            for (i in this.indices)
                this[i] -= other
        }
        else -> {
            val left = this.asDNArray()
            for (index in this.multiIndices)
                left[index] -= other
        }
    }
}

/**
 * Create a new array as product of [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.times(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret *= other
    return ret
}

/**
 * Multiply [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
    when {
        this.consistent && other.consistent -> {
            this.data *= (other.data as MemoryView)
        }
        this.consistent && !other.consistent -> {
            val iterRight = other.iterator()
            for (i in this.indices)
                this.data[i] *= iterRight.next()
        }
        else -> {
            val left = this.asDNArray()
            val iterRight = other.iterator()
            for (index in this.multiIndices)
                left[index] *= iterRight.next()
        }
    }
}

/**
 * Multiply [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: T) {
    when {
        this.consistent -> {
            this.data *= other
        }
        dim.d == 1 -> {
            this as MutableMultiArray<T, D1>
            for (i in this.indices)
                this[i] *= other
        }
        else -> {
            val left = this.asDNArray()
            for (index in this.multiIndices)
                left[index] *= other
        }
    }
}

/**
 * Create a new array as division of [this] by [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): NDArray<T, D> {
    requireArraySizes(this.size, other.size)
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.div(other: T): NDArray<T, D> {
    val ret = if (this.consistent) (this as NDArray).clone() else (this as NDArray).deepCopy()
    ret /= other
    return ret
}

/**
 * Divide [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.divAssign(other: MultiArray<T, D>) {
    requireArraySizes(this.size, other.size)
    when {
        this.consistent && other.consistent -> {
            this.data /= (other.data as MemoryView)
        }
        this.consistent && !other.consistent -> {
            val iterRight = other.iterator()
            for (i in this.indices)
                this.data[i] /= iterRight.next()
        }
        else -> {
            val left = this.asDNArray()
            val iterRight = other.iterator()
            for (index in this.multiIndices)
                left[index] /= iterRight.next()
        }
    }
}

/**
 * Divide by [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.divAssign(other: T) {
    when {
        this.consistent -> {
            this.data /= other
        }
        dim.d == 1 -> {
            this as MutableMultiArray<T, D1>
            for (i in this.indices)
                this[i] /= other
        }
        else -> {
            val left = this.asDNArray()
            for (index in this.multiIndices)
                left[index] /= other
        }
    }
}

//TODO(create module utils)
@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
/*internal*/ public inline operator fun <T : Number> Number.plus(other: T): T = when (this) {
    is Byte -> (this.toByte() + other.toByte()).toByte()
    is Short -> (this.toShort() + other.toShort()).toShort()
    is Int -> (this.toInt() + other.toInt())
    is Long -> (this.toLong() + other.toLong())
    is Float -> (this.toFloat() + other.toFloat())
    is Double -> (this.toDouble() + other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
/*internal*/ public inline operator fun <T : Number> Number.minus(other: T): T = when (this) {
    is Byte -> (this.toByte() - other.toByte()).toByte()
    is Short -> (this.toShort() - other.toShort()).toShort()
    is Int -> (this.toInt() - other.toInt())
    is Long -> (this.toLong() - other.toLong())
    is Float -> (this.toFloat() - other.toFloat())
    is Double -> (this.toDouble() - other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
/*internal*/ public inline operator fun <T : Number> Number.times(other: T): T = when (this) {
    is Byte -> (this.toByte() * other.toByte()).toByte()
    is Short -> (this.toShort() * other.toShort()).toShort()
    is Int -> (this.toInt() * other.toInt())
    is Long -> (this.toLong() * other.toLong())
    is Float -> (this.toFloat() * other.toFloat())
    is Double -> (this.toDouble() * other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
/*internal*/ public inline operator fun <T : Number> Number.div(other: T): T = when (this) {
    is Byte -> (this.toByte() / other.toByte()).toByte()
    is Short -> (this.toShort() / other.toShort()).toShort()
    is Int -> (this.toInt() / other.toInt())
    is Long -> (this.toLong() / other.toLong())
    is Float -> (this.toFloat() / other.toFloat())
    is Double -> (this.toDouble() / other.toDouble())
    else -> throw Exception("Type not defined.")
} as T