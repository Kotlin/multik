package org.jetbrains.multik.core


//TODO(arithmetic function of dimension)
/**
 * Create a new array as the sum of [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): Ndarray<T, D> {
    requireArraySizes(this.size, other.size)
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] + other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] + right[index]
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.plus(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] + other
    } else {
        val left = this.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] + other
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Add [other] to [this]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: MultiArray<T, D>): Unit {
    requireArraySizes(this.size, other.size)
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            this[i] += other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        for (index in this.multiIndices)
            left[index] += right[index]
    }
}


/**
 * Add [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: T): Unit {
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        for (i in this.indices)
            this[i] += other
    } else {
        val left = this.asDNArray()
        for (index in this.multiIndices)
            left[index] += other
    }
}


/**
 * Create a new array as difference between [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): Ndarray<T, D> {
    requireArraySizes(this.size, other.size)
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] - other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] - right[index]
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.minus(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] - other
    } else {
        val left = this.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] - other
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Subtract [other] from [this]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: MultiArray<T, D>): Unit {
    requireArraySizes(this.size, other.size)
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            this[i] -= other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        for (index in this.multiIndices)
            left[index] -= right[index]
    }
}

/**
 * Subtract [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: T): Unit {
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        for (i in this.indices)
            this[i] -= other
    } else {
        val left = this.asDNArray()
        for (index in this.multiIndices)
            left[index] -= other
    }
}

/**
 * Create a new array as product of [this] and [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): Ndarray<T, D> {
    requireArraySizes(this.size, other.size)
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] * other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] * right[index]
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.times(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] * other
    } else {
        val left = this.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] * other
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Multiply [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: MultiArray<T, D>): Unit {
    requireArraySizes(this.size, other.size)
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            this[i] *= other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        for (index in this.multiIndices)
            left[index] *= right[index]
    }
}

/**
 * Multiply [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: T): Unit {
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        for (i in this.indices)
            this[i] *= other
    } else {
        val left = this.asDNArray()
        for (index in this.multiIndices)
            left[index] *= other
    }
}

/**
 * Create a new array as division of [this] by [other].
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): Ndarray<T, D> {
    requireArraySizes(this.size, other.size)
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] / other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] / right[index]
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

public operator fun <T : Number, D : Dimension> MultiArray<T, D>.div(other: T): Ndarray<T, D> {
    val data = initMemoryView<T>(size, dtype)
    if (this.dim.d == 1) {
        this as MultiArray<T, D1>
        for (i in this.indices)
            data[i] = this[i] / other
    } else {
        val left = this.asDNArray()
        var i = 0
        for (index in this.multiIndices)
            data[i++] = left[index] / other
    }
    return Ndarray<T, D>(data, shape = shape, dtype = dtype, dim = dim)
}

/**
 * Divide [this] by [other]. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.divAssign(other: MultiArray<T, D>): Unit {
    requireArraySizes(this.size, other.size)
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        other as MultiArray<T, D1>
        for (i in this.indices)
            this[i] /= other[i]
    } else {
        val left = this.asDNArray()
        val right = other.asDNArray()
        for (index in this.multiIndices)
            left[index] /= right[index]
    }
}

/**
 * Divide by [other] element-wise. Inplace operator.
 */
public operator fun <T : Number, D : Dimension> MutableMultiArray<T, D>.divAssign(other: T): Unit {
    if (this.dim.d == 1) {
        this as MutableMultiArray<T, D1>
        for (i in this.indices)
            this[i] /= other
    } else {
        val left = this.asDNArray()
        for (index in this.multiIndices)
            left[index] /= other
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