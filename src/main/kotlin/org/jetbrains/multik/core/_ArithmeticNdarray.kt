package org.jetbrains.multik.core

import org.jetbrains.multik.jni.Basic

/**
 * Create a new array as the sum of [this] and [other].
 *
 * TODO(assert shape and dim)
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plus(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    for (i in indices)
        data.put(i, this[i] + other[i])
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, shape, size, dtype, dim)
}

/**
 * Add [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plusAssign(other: T): Unit {
    for (i in indices)
        this[i] = this[i] + other
}

/**
 * Add [other] to [this]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plusAssign(other: Ndarray<T, D>): Unit {
    for (i in indices)
        this[i] = this[i] + other[i]
}

/**
 * Create a new array as difference between [this] and [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minus(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    for (i in indices) {
        data.put(i, this[i] - other[i])
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, shape, size, dtype, dim)
}

/**
 * Subtract [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minusAssign(other: T): Unit {
    for (i in indices)
        this[i] = this[i] - other
}

/**
 * Subtract [other] from [this]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minusAssign(other: Ndarray<T, D>): Unit {
    for (i in indices)
        this[i] = this[i] - other[i]
}

/**
 * Create a new array as product of [this] and [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.times(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    for (i in indices) {
        data.put(i, this[i] * other[i])
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, shape, size, dtype, dim)
}

/**
 * Multiply [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.timesAssign(other: T): Unit {
    for (i in indices)
        this[i] = this[i] * other
}

/**
 * Multiply [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.timesAssign(other: Ndarray<T, D>): Unit {
    for (i in indices)
        this[i] = this[i] * other[i]
}

/**
 * Create a new array as division of [this] by [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.div(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    for (i in indices) {
        data.put(i, this[i] / other[i])
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, shape, size, dtype, dim)
}

/**
 * Divide by [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.divAssign(other: T): Unit {
    for (i in indices)
        this[i] = this[i] / other
}

/**
 * Divide [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.divAssign(other: Ndarray<T, D>): Unit {
    for (i in indices)
        this[i] = this[i] / other[i]
}

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
private inline operator fun <T : Number> Number.plus(other: T): T = when (this) {
    is Byte -> (this.toByte() + other.toByte()).toByte()
    is Short -> (this.toShort() + other.toShort()).toShort()
    is Int -> (this.toInt() + other.toInt())
    is Long -> (this.toLong() + other.toLong())
    is Float -> (this.toFloat() + other.toFloat())
    is Double -> (this.toDouble() + other.toDouble())
    else -> throw Exception("")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
private inline operator fun <T : Number> Number.minus(other: T): T = when (this) {
    is Byte -> (this.toByte() - other.toByte()).toByte()
    is Short -> (this.toShort() - other.toShort()).toShort()
    is Int -> (this.toInt() - other.toInt())
    is Long -> (this.toLong() - other.toLong())
    is Float -> (this.toFloat() - other.toFloat())
    is Double -> (this.toDouble() - other.toDouble())
    else -> throw Exception("")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
private inline operator fun <T : Number> Number.times(other: T): T = when (this) {
    is Byte -> (this.toByte() * other.toByte()).toByte()
    is Short -> (this.toShort() * other.toShort()).toShort()
    is Int -> (this.toInt() * other.toInt())
    is Long -> (this.toLong() * other.toLong())
    is Float -> (this.toFloat() * other.toFloat())
    is Double -> (this.toDouble() * other.toDouble())
    else -> throw Exception("")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
private inline operator fun <T : Number> Number.div(other: T): T = when (this) {
    is Byte -> (this.toByte() / other.toByte()).toByte()
    is Short -> (this.toShort() / other.toShort()).toShort()
    is Int -> (this.toInt() / other.toInt())
    is Long -> (this.toLong() / other.toLong())
    is Float -> (this.toFloat() / other.toFloat())
    is Double -> (this.toDouble() / other.toDouble())
    else -> throw Exception("")
} as T