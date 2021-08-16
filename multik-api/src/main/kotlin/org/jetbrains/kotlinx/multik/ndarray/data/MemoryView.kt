/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.complex.*

/**
 * View for storing data in a [NDArray] and working them in a uniform style.
 *
 * @property data one of the primitive arrays.
 */
public interface ImmutableMemoryView<T> : Iterable<T> {
    public val data: Any
    public var size: Int

    /**
     * Returns the value at [index].
     *
     * Note: Indexing takes place according to the initial data, so if you did any manipulations with the ndarray
     * (ex. reshape), then `get` from the ndarray with the same index will return another value.
     */
    public operator fun get(index: Int): T

    /**
     * [data] iterator
     */
    public override fun iterator(): Iterator<T>

    /**
     * Returns a new instance with a copied primitive array.
     */
    public fun copyOf(): ImmutableMemoryView<T>

    public fun copyInto(
        destination: ImmutableMemoryView<T>, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size
    ): ImmutableMemoryView<T>

    /**
     * Returns [ByteArray] if it is [MemoryViewByteArray].
     */
    public fun getByteArray(): ByteArray

    /**
     * Returns [ShortArray] if it is [MemoryViewShortArray].
     */
    public fun getShortArray(): ShortArray

    /**
     * Returns [IntArray] if it is [MemoryViewIntArray].
     */
    public fun getIntArray(): IntArray

    /**
     * Returns [LongArray] if it is [MemoryViewLongArray].
     */
    public fun getLongArray(): LongArray

    /**
     * Returns [FloatArray] if it is [MemoryViewFloatArray].
     *
     * Note: For [MemoryViewComplexFloatArray], an array will be returned storing real and imaginary parts continuously.
     */
    public fun getFloatArray(): FloatArray

    /**
     * Returns [DoubleArray].
     *
     * Note: For [MemoryViewComplexDoubleArray], an array will be returned storing real and imaginary parts continuously.
     */
    public fun getDoubleArray(): DoubleArray

    /**
     * Returns [ComplexFloatArray] if it is [MemoryViewFloatArray].
     */
    public fun getComplexFloatArray(): ComplexFloatArray

    /**
     * Returns [ComplexDoubleArray] if it is [MemoryViewComplexDoubleArray].
     */
    public fun getComplexDoubleArray(): ComplexDoubleArray
}

/**
 * Extends [ImmutableMemoryView].
 *
 * @property size number of elements in [data].
 * @property indices indices of [data].
 * @property lastIndex last index in [data].
 */
public abstract class MemoryView<T> : ImmutableMemoryView<T> {
    public abstract var indices: IntRange

    public abstract var lastIndex: Int

    /**
     * Replaces the element at the given [index] with the specified [value].
     *
     * Note: Indexing takes place according to the initial data.
     */
    public abstract operator fun set(index: Int, value: T)

    public abstract override fun copyOf(): MemoryView<T>

    public abstract override fun copyInto(
        destination: ImmutableMemoryView<T>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryView<T>

    public override fun getByteArray(): ByteArray = throw UnsupportedOperationException()

    public override fun getShortArray(): ShortArray = throw UnsupportedOperationException()

    public override fun getIntArray(): IntArray = throw UnsupportedOperationException()

    public override fun getLongArray(): LongArray = throw UnsupportedOperationException()

    public override fun getFloatArray(): FloatArray = throw UnsupportedOperationException()

    public override fun getDoubleArray(): DoubleArray = throw UnsupportedOperationException()

    public override fun getComplexFloatArray(): ComplexFloatArray = throw UnsupportedOperationException()

    public override fun getComplexDoubleArray(): ComplexDoubleArray = throw UnsupportedOperationException()

    public operator fun plusAssign(other: MemoryView<T>) {
        when {
            this is MemoryViewFloatArray && other is MemoryViewFloatArray -> this += other
            this is MemoryViewIntArray && other is MemoryViewIntArray -> this += other
            this is MemoryViewDoubleArray && other is MemoryViewDoubleArray -> this += other
            this is MemoryViewLongArray && other is MemoryViewLongArray -> this += other
            this is MemoryViewShortArray && other is MemoryViewShortArray -> this += other
            this is MemoryViewByteArray && other is MemoryViewByteArray -> this += other
            this is MemoryViewComplexFloatArray && other is MemoryViewComplexFloatArray -> this += other
            this is MemoryViewComplexDoubleArray && other is MemoryViewComplexDoubleArray -> this += other
        }
    }

    public open operator fun plusAssign(other: T) {
        when {
            this is MemoryViewFloatArray && other is Float -> this += other
            this is MemoryViewIntArray && other is Int -> this += other
            this is MemoryViewDoubleArray && other is Double -> this += other
            this is MemoryViewLongArray && other is Long -> this += other
            this is MemoryViewShortArray && other is Short -> this += other
            this is MemoryViewByteArray && other is Byte -> this += other
            this is MemoryViewComplexFloatArray && other is ComplexFloat -> this += other
            this is MemoryViewComplexDoubleArray && other is ComplexDouble -> this += other
        }
    }

    public operator fun minusAssign(other: MemoryView<T>) {
        when {
            this is MemoryViewFloatArray && other is MemoryViewFloatArray -> this -= other
            this is MemoryViewIntArray && other is MemoryViewIntArray -> this -= other
            this is MemoryViewDoubleArray && other is MemoryViewDoubleArray -> this -= other
            this is MemoryViewLongArray && other is MemoryViewLongArray -> this -= other
            this is MemoryViewShortArray && other is MemoryViewShortArray -> this -= other
            this is MemoryViewByteArray && other is MemoryViewByteArray -> this -= other
            this is MemoryViewComplexFloatArray && other is MemoryViewComplexFloatArray -> this -= other
            this is MemoryViewComplexDoubleArray && other is MemoryViewComplexDoubleArray -> this -= other
        }
    }

    public open operator fun minusAssign(other: T) {
        when {
            this is MemoryViewFloatArray && other is Float -> this -= other
            this is MemoryViewIntArray && other is Int -> this -= other
            this is MemoryViewDoubleArray && other is Double -> this -= other
            this is MemoryViewLongArray && other is Long -> this -= other
            this is MemoryViewShortArray && other is Short -> this -= other
            this is MemoryViewByteArray && other is Byte -> this -= other
            this is MemoryViewComplexFloatArray && other is ComplexFloat -> this -= other
            this is MemoryViewComplexDoubleArray && other is ComplexDouble -> this -= other
        }
    }

    public operator fun timesAssign(other: MemoryView<T>) {
        when {
            this is MemoryViewFloatArray && other is MemoryViewFloatArray -> this *= other
            this is MemoryViewIntArray && other is MemoryViewIntArray -> this *= other
            this is MemoryViewDoubleArray && other is MemoryViewDoubleArray -> this *= other
            this is MemoryViewLongArray && other is MemoryViewLongArray -> this *= other
            this is MemoryViewShortArray && other is MemoryViewShortArray -> this *= other
            this is MemoryViewByteArray && other is MemoryViewByteArray -> this *= other
            this is MemoryViewComplexFloatArray && other is MemoryViewComplexFloatArray -> this *= other
            this is MemoryViewComplexDoubleArray && other is MemoryViewComplexDoubleArray -> this *= other
        }
    }

    public open operator fun timesAssign(other: T) {
        when {
            this is MemoryViewFloatArray && other is Float -> this *= other
            this is MemoryViewIntArray && other is Int -> this *= other
            this is MemoryViewDoubleArray && other is Double -> this *= other
            this is MemoryViewLongArray && other is Long -> this *= other
            this is MemoryViewShortArray && other is Short -> this *= other
            this is MemoryViewByteArray && other is Byte -> this *= other
            this is MemoryViewComplexFloatArray && other is ComplexFloat -> this *= other
            this is MemoryViewComplexDoubleArray && other is ComplexDouble -> this *= other
        }
    }

    public operator fun divAssign(other: MemoryView<T>) {
        when {
            this is MemoryViewFloatArray && other is MemoryViewFloatArray -> this /= other
            this is MemoryViewIntArray && other is MemoryViewIntArray -> this /= other
            this is MemoryViewDoubleArray && other is MemoryViewDoubleArray -> this /= other
            this is MemoryViewLongArray && other is MemoryViewLongArray -> this /= other
            this is MemoryViewShortArray && other is MemoryViewShortArray -> this /= other
            this is MemoryViewByteArray && other is MemoryViewByteArray -> this /= other
            this is MemoryViewComplexFloatArray && other is MemoryViewComplexFloatArray -> this /= other
            this is MemoryViewComplexDoubleArray && other is MemoryViewComplexDoubleArray -> this /= other
        }
    }

    public open operator fun divAssign(other: T) {
        when {
            this is MemoryViewFloatArray && other is Float -> this /= other
            this is MemoryViewIntArray && other is Int -> this /= other
            this is MemoryViewDoubleArray && other is Double -> this /= other
            this is MemoryViewLongArray && other is Long -> this /= other
            this is MemoryViewShortArray && other is Short -> this /= other
            this is MemoryViewByteArray && other is Byte -> this /= other
            this is MemoryViewComplexFloatArray && other is ComplexFloat -> this /= other
            this is MemoryViewComplexDoubleArray && other is ComplexDouble -> this /= other
        }
    }
}

/**
 * View for [ByteArray].
 */
public class MemoryViewByteArray(override val data: ByteArray) : MemoryView<Byte>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Byte = data[index]

    override fun set(index: Int, value: Byte) {
        data[index] = value
    }

    override fun getByteArray(): ByteArray = data

    override fun iterator(): Iterator<Byte> = data.iterator()

    override fun copyOf(): MemoryView<Byte> = MemoryViewByteArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Byte>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewByteArray {
        val retArray = this.data.copyInto(destination.getByteArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewByteArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewByteArray -> false
        size != other.size -> false
        else -> (0 until size).all { this.data[it] == other.data[it] }
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewByteArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] + other.data[i]).toByte()
        }
    }

    public override operator fun plusAssign(other: Byte) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] + other).toByte()
        }
    }

    public operator fun minusAssign(other: MemoryViewByteArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] - other.data[i]).toByte()
        }
    }

    public override operator fun minusAssign(other: Byte) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] - other).toByte()
        }
    }

    public operator fun timesAssign(other: MemoryViewByteArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] * other.data[i]).toByte()
        }
    }

    public override operator fun timesAssign(other: Byte) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] * other).toByte()
        }
    }

    public operator fun divAssign(other: MemoryViewByteArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] / other.data[i]).toByte()
        }
    }

    public override operator fun divAssign(other: Byte) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] / other).toByte()
        }
    }
}

/**
 * View for [ShortArray].
 */
public class MemoryViewShortArray(override val data: ShortArray) : MemoryView<Short>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Short = data[index]

    override fun set(index: Int, value: Short) {
        data[index] = value
    }

    override fun getShortArray(): ShortArray = data

    override fun iterator(): Iterator<Short> = data.iterator()

    override fun copyOf(): MemoryView<Short> = MemoryViewShortArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Short>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewShortArray {
        val retArray = this.data.copyInto(destination.getShortArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewShortArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewShortArray -> false
        size != other.size -> false
        else -> this.data.contentEquals(other.data)
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewShortArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] + other.data[i]).toShort()
        }
    }

    public override operator fun plusAssign(other: Short) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] + other).toShort()
        }
    }

    public operator fun minusAssign(other: MemoryViewShortArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] - other.data[i]).toShort()
        }
    }

    public override operator fun minusAssign(other: Short) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] - other).toShort()
        }
    }

    public operator fun timesAssign(other: MemoryViewShortArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] * other.data[i]).toShort()
        }
    }

    public override operator fun timesAssign(other: Short) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] * other).toShort()
        }
    }

    public operator fun divAssign(other: MemoryViewShortArray) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] / other.data[i]).toShort()
        }
    }

    public override operator fun divAssign(other: Short) {
        for (i in this.indices) {
            this.data[i] = (this.data[i] / other).toShort()
        }
    }
}

/**
 * View for [IntArray].
 */
public class MemoryViewIntArray(override val data: IntArray) : MemoryView<Int>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Int = data[index]

    override fun set(index: Int, value: Int) {
        data[index] = value
    }

    override fun getIntArray(): IntArray = data

    override fun iterator(): Iterator<Int> = data.iterator()

    override fun copyOf(): MemoryView<Int> = MemoryViewIntArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Int>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewIntArray {
        val retArray = this.data.copyInto(destination.getIntArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewIntArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewIntArray -> false
        size != other.size -> false
        else -> (0 until size).all { this.data[it] == other.data[it] }
    }


    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewIntArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: Int) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewIntArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: Int) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewIntArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: Int) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewIntArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: Int) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }
}

/**
 * View for [LongArray].
 */
public class MemoryViewLongArray(override val data: LongArray) : MemoryView<Long>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Long = data[index]

    override fun set(index: Int, value: Long) {
        data[index] = value
    }

    override fun getLongArray(): LongArray = data

    override fun iterator(): Iterator<Long> = data.iterator()

    override fun copyOf(): MemoryView<Long> = MemoryViewLongArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Long>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewLongArray {
        val retArray = this.data.copyInto(destination.getLongArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewLongArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewLongArray -> false
        size != other.size -> false
        else -> this.data.contentEquals(other.data)
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewLongArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: Long) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewLongArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: Long) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewLongArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: Long) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewLongArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: Long) {
        for (i in this.indices) {
            this.data[i] /= other
        }
    }
}

/**
 * View for [FloatArray].
 */
public class MemoryViewFloatArray(override val data: FloatArray) : MemoryView<Float>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Float = data[index]

    override fun set(index: Int, value: Float) {
        data[index] = value
    }

    override fun getFloatArray(): FloatArray = data

    override fun iterator(): Iterator<Float> = data.iterator()

    override fun copyOf(): MemoryView<Float> = MemoryViewFloatArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Float>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewFloatArray {
        val retArray = this.data.copyInto(destination.getFloatArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewFloatArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewFloatArray -> false
        size != other.size -> false
        else -> this.data.contentEquals(other.data)
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewFloatArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: Float) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewFloatArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: Float) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewFloatArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: Float) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewFloatArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: Float) {
        for (i in this.indices) {
            this.data[i] /= other
        }
    }
}

/**
 * View for [DoubleArray].
 */
public class MemoryViewDoubleArray(override val data: DoubleArray) : MemoryView<Double>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Double = data[index]

    override fun set(index: Int, value: Double) {
        data[index] = value
    }

    override fun getDoubleArray(): DoubleArray = data

    override fun iterator(): Iterator<Double> = data.iterator()

    override fun copyOf(): MemoryView<Double> = MemoryViewDoubleArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<Double>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewDoubleArray {
        val retArray = this.data.copyInto(destination.getDoubleArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewDoubleArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewDoubleArray -> false
        size != other.size -> false
        else -> this.data.contentEquals(other.data)
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewDoubleArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: Double) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewDoubleArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: Double) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewDoubleArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: Double) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewDoubleArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: Double) {
        for (i in this.indices) {
            this.data[i] /= other
        }
    }
}

/**
 * View for [ComplexFloatArray].
 */
public class MemoryViewComplexFloatArray(override val data: ComplexFloatArray) : MemoryView<ComplexFloat>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): ComplexFloat = data[index]

    override fun set(index: Int, value: ComplexFloat) {
        data[index] = value
    }

    override fun getFloatArray(): FloatArray = data.getFlatArray()

    override fun getComplexFloatArray(): ComplexFloatArray = data

    override fun iterator(): Iterator<ComplexFloat> = data.iterator()

    override fun copyOf(): MemoryView<ComplexFloat> = MemoryViewComplexFloatArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<ComplexFloat>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewComplexFloatArray {
        val retArray = this.data.copyInto(destination.getComplexFloatArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewComplexFloatArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewComplexFloatArray -> false
        size != other.size -> false
        else -> this.data == other.data
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewComplexFloatArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: ComplexFloat) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewComplexFloatArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: ComplexFloat) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewComplexFloatArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: ComplexFloat) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewComplexFloatArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: ComplexFloat) {
        for (i in this.indices) {
            this.data[i] /= other
        }
    }
}

/**
 * View for [ComplexDoubleArray].
 */
public class MemoryViewComplexDoubleArray(override val data: ComplexDoubleArray) : MemoryView<ComplexDouble>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): ComplexDouble = data[index]

    override fun set(index: Int, value: ComplexDouble) {
        data[index] = value
    }

    override fun getDoubleArray(): DoubleArray = data.getFlatArray()

    override fun getComplexDoubleArray(): ComplexDoubleArray = data

    override fun iterator(): Iterator<ComplexDouble> = data.iterator()

    override fun copyOf(): MemoryView<ComplexDouble> = MemoryViewComplexDoubleArray(data.copyOf())

    override fun copyInto(
        destination: ImmutableMemoryView<ComplexDouble>, destinationOffset: Int, startIndex: Int, endIndex: Int
    ): MemoryViewComplexDoubleArray {
        val retArray = this.data.copyInto(destination.getComplexDoubleArray(), destinationOffset, startIndex, endIndex)
        return MemoryViewComplexDoubleArray(retArray)
    }

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is MemoryViewComplexDoubleArray -> false
        size != other.size -> false
        else -> this.data == other.data
    }

    override fun hashCode(): Int =
        (0 until size).fold(1) { acc, r ->
            31 * acc + data[r].hashCode()
        }

    public operator fun plusAssign(other: MemoryViewComplexDoubleArray) {
        for (i in this.indices) {
            this.data[i] += other.data[i]
        }
    }

    public override operator fun plusAssign(other: ComplexDouble) {
        for (i in this.indices) {
            this.data[i] += other
        }
    }

    public operator fun minusAssign(other: MemoryViewComplexDoubleArray) {
        for (i in this.indices) {
            this.data[i] -= other.data[i]
        }
    }

    public override operator fun minusAssign(other: ComplexDouble) {
        for (i in this.indices) {
            this.data[i] -= other
        }
    }

    public operator fun timesAssign(other: MemoryViewComplexDoubleArray) {
        for (i in this.indices) {
            this.data[i] *= other.data[i]
        }
    }

    public override operator fun timesAssign(other: ComplexDouble) {
        for (i in this.indices) {
            this.data[i] *= other
        }
    }

    public operator fun divAssign(other: MemoryViewComplexDoubleArray) {
        for (i in this.indices) {
            this.data[i] /= other.data[i]
        }
    }

    public override operator fun divAssign(other: ComplexDouble) {
        for (i in this.indices) {
            this.data[i] /= other
        }
    }
}

/**
 * Creates a [MemoryView] based [size] and [dataType].
 */
public fun <T> initMemoryView(size: Int, dataType: DataType): MemoryView<T> {
    val t = when (dataType) {
        DataType.ByteDataType -> MemoryViewByteArray(ByteArray(size))
        DataType.ShortDataType -> MemoryViewShortArray(ShortArray(size))
        DataType.IntDataType -> MemoryViewIntArray(IntArray(size))
        DataType.LongDataType -> MemoryViewLongArray(LongArray(size))
        DataType.FloatDataType -> MemoryViewFloatArray(FloatArray(size))
        DataType.DoubleDataType -> MemoryViewDoubleArray(DoubleArray(size))
        DataType.ComplexFloatDataType -> MemoryViewComplexFloatArray(ComplexFloatArray(size))
        DataType.ComplexDoubleDataType -> MemoryViewComplexDoubleArray(ComplexDoubleArray(size))
    }
    @Suppress("UNCHECKED_CAST")
    return t as MemoryView<T>
}

/**
 * Create a [MemoryView] based [size] and [dataType], where each elements will be initialized according
 * to the given [init] function.
 */
@Suppress("UNCHECKED_CAST")
public fun <T> initMemoryView(size: Int, dataType: DataType, init: (Int) -> T): MemoryView<T> {
    val t = when (dataType) {
        DataType.ByteDataType -> MemoryViewByteArray(ByteArray(size, init as (Int) -> Byte))
        DataType.ShortDataType -> MemoryViewShortArray(ShortArray(size, init as (Int) -> Short))
        DataType.IntDataType -> MemoryViewIntArray(IntArray(size, init as (Int) -> Int))
        DataType.LongDataType -> MemoryViewLongArray(LongArray(size, init as (Int) -> Long))
        DataType.FloatDataType -> MemoryViewFloatArray(FloatArray(size, init as (Int) -> Float))
        DataType.DoubleDataType -> MemoryViewDoubleArray(DoubleArray(size, init as (Int) -> Double))
        DataType.ComplexFloatDataType -> MemoryViewComplexFloatArray(ComplexFloatArray(size, init as (Int) -> ComplexFloat))
        DataType.ComplexDoubleDataType -> MemoryViewComplexDoubleArray(ComplexDoubleArray(size, init as (Int) -> ComplexDouble))
    }
    return t as MemoryView<T>
}

@Suppress("UNCHECKED_CAST")
public fun <T> List<T>.toViewPrimitiveArray(dataType: DataType): MemoryView<T> {
    val t = when (dataType) {
        DataType.ByteDataType -> MemoryViewByteArray((this as List<Byte>).toByteArray())
        DataType.ShortDataType -> MemoryViewShortArray((this as List<Short>).toShortArray())
        DataType.IntDataType -> MemoryViewIntArray((this as List<Int>).toIntArray())
        DataType.LongDataType -> MemoryViewLongArray((this as List<Long>).toLongArray())
        DataType.FloatDataType -> MemoryViewFloatArray((this as List<Float>).toFloatArray())
        DataType.DoubleDataType -> MemoryViewDoubleArray((this as List<Double>).toDoubleArray())
        DataType.ComplexFloatDataType -> MemoryViewComplexFloatArray((this as List<ComplexFloat>).toComplexFloatArray())
        DataType.ComplexDoubleDataType -> MemoryViewComplexDoubleArray((this as List<ComplexDouble>).toComplexDoubleArray())
    }
    return t as MemoryView<T>
}