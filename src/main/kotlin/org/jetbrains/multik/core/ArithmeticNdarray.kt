package org.jetbrains.multik.core

/**
 * Create a new array as the sum of [this] and [other].
 *
 * TODO(assert shape and dim)
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plus(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] + other[i]
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] + other[index]
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : DN> Ndarray<T, D>.plus(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] + other
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] + other
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Add [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plusAssign(other: T): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] += other
    } else {
        for (index in this.multiIndices)
            this[index] += other
    }
}


//todo (view and strides?)
//todo (mismatch shape)
/**
 * Add [other] to [this]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.plusAssign(other: Ndarray<T, D>): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] += other[i]
    } else {
        for (index in this.multiIndices)
            this[index] += other[index]
    }
}

/**
 * Create a new array as difference between [this] and [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minus(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] - other[i]
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] - other[index]
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : DN> Ndarray<T, D>.minus(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] - other
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] - other
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Subtract [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minusAssign(other: T): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] -= other
    } else {
        for (index in this.multiIndices)
            this[index] -= other
    }
}

/**
 * Subtract [other] from [this]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.minusAssign(other: Ndarray<T, D>): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] -= other[i]
    } else {
        for (index in this.multiIndices)
            this[index] -= other[index]
    }
}

/**
 * Create a new array as product of [this] and [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.times(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] * other[i]
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] * other[index]
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : DN> Ndarray<T, D>.times(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] * other
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] * other
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Multiply [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.timesAssign(other: T): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] *= other
    } else {
        for (index in this.multiIndices)
            this[index] *= other
    }
}

/**
 * Multiply [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.timesAssign(other: Ndarray<T, D>): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] *= other[i]
    } else {
        for (index in this.multiIndices)
            this[index] *= other[index]
    }
}

/**
 * Create a new array as division of [this] by [other].
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.div(other: Ndarray<T, D>): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] / other[i]
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] / other[index]
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : DN> Ndarray<T, D>.div(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        for (i in this.indices)
            data[i] = this[i] / other
    } else {
        var i = 0
        for (index in this.multiIndices)
            data[i++] = this[index] / other
    }
    return initNdarray(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Divide by [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.divAssign(other: T): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] /= other
    } else {
        for (index in this.multiIndices)
            this[index] /= other
    }
}

/**
 * Divide [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : DN> Ndarray<T, D>.divAssign(other: Ndarray<T, D>): Unit {
    if (this is D1Array) {
        for (i in this.indices)
            this[i] /= other[i]
    } else {
        for (index in this.multiIndices)
            this[index] /= other[index]
    }
}

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
internal inline operator fun <T : Number> Number.plus(other: T): T = when (this) {
    is Byte -> (this.toByte() + other.toByte()).toByte()
    is Short -> (this.toShort() + other.toShort()).toShort()
    is Int -> (this.toInt() + other.toInt())
    is Long -> (this.toLong() + other.toLong())
    is Float -> (this.toFloat() + other.toFloat())
    is Double -> (this.toDouble() + other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
internal inline operator fun <T : Number> Number.minus(other: T): T = when (this) {
    is Byte -> (this.toByte() - other.toByte()).toByte()
    is Short -> (this.toShort() - other.toShort()).toShort()
    is Int -> (this.toInt() - other.toInt())
    is Long -> (this.toLong() - other.toLong())
    is Float -> (this.toFloat() - other.toFloat())
    is Double -> (this.toDouble() - other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
internal inline operator fun <T : Number> Number.times(other: T): T = when (this) {
    is Byte -> (this.toByte() * other.toByte()).toByte()
    is Short -> (this.toShort() * other.toShort()).toShort()
    is Int -> (this.toInt() * other.toInt())
    is Long -> (this.toLong() * other.toLong())
    is Float -> (this.toFloat() * other.toFloat())
    is Double -> (this.toDouble() * other.toDouble())
    else -> throw Exception("Type not defined.")
} as T

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST", "IMPLICIT_CAST_TO_ANY")
internal inline operator fun <T : Number> Number.div(other: T): T = when (this) {
    is Byte -> (this.toByte() / other.toByte()).toByte()
    is Short -> (this.toShort() / other.toShort()).toShort()
    is Int -> (this.toInt() / other.toInt())
    is Long -> (this.toLong() / other.toLong())
    is Float -> (this.toFloat() / other.toFloat())
    is Double -> (this.toDouble() / other.toDouble())
    else -> throw Exception("Type not defined.")
} as T