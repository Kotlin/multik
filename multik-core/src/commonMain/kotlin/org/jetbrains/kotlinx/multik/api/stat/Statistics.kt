package org.jetbrains.kotlinx.multik.api.stat

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName

/**
 * Statistical operations interface, accessible via `mk.stat`.
 *
 * Provides descriptive statistics: [median], [average], and [mean] (with axis-based variants).
 * All results are [Double]-precision.
 *
 * @see [org.jetbrains.kotlinx.multik.api.math.Math] for aggregation functions (sum, min, max).
 */
public interface Statistics {

    /**
     * Returns the median of all elements in [a], or `null` if the array is empty.
     *
     * @param a the input array.
     * @return the median value as [Double], or `null`.
     */
    public fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double?

    /**
     * Returns the weighted average of [a] using the given [weights].
     *
     * When [weights] is `null`, computes the unweighted average (equivalent to [mean]).
     *
     * @param a the input array.
     * @param weights optional weights array with the same shape as [a].
     * @return the weighted average as [Double].
     * @see [mean] for the arithmetic mean.
     */
    public fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>? = null): Double

    /**
     * Returns the arithmetic mean of all elements in [a].
     *
     * @param a the input array.
     * @return the arithmetic mean as [Double].
     * @see [average] for weighted average.
     */
    public fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double

    /**
     * Returns the arithmetic mean along the given [axis], reducing the rank by one.
     *
     * @param a the input array.
     * @param axis the axis to reduce.
     * @return an [NDArray] of [Double] means with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O>

    /** Dimension-specific [mean] along [axis] for 2D arrays. */
    public fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1>

    /** Dimension-specific [mean] along [axis] for 3D arrays. */
    public fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2>

    /** Dimension-specific [mean] along [axis] for 4D arrays. */
    public fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3>

    /** Dimension-specific [mean] along [axis] for N-dimensional arrays. */
    public fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, DN>
}

/**
 * Returns a new array with the absolute value of each element in [a].
 *
 * @param a the input [Byte] array.
 * @return a new [NDArray] of the same shape with element-wise absolute values.
 */
@JvmName("absByte")
public fun <D : Dimension> abs(a: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = initMemoryView<Byte>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absByte(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Returns a new array with the absolute value of each [Short] element in [a]. */
@JvmName("absShort")
public fun <D : Dimension> abs(a: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = initMemoryView<Short>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absShort(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Returns a new array with the absolute value of each [Int] element in [a]. */
@JvmName("absInt")
public fun <D : Dimension> abs(a: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = initMemoryView<Int>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Returns a new array with the absolute value of each [Long] element in [a]. */
@JvmName("absLong")
public fun <D : Dimension> abs(a: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = initMemoryView<Long>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Returns a new array with the absolute value of each [Float] element in [a]. */
@JvmName("absFloat")
public fun <D : Dimension> abs(a: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = initMemoryView<Float>(a.size)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Returns a new array with the absolute value of each [Double] element in [a]. */
@JvmName("absDouble")
public fun <D : Dimension> abs(a: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = initMemoryView<Double>(a.size)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/**
 * Returns a new array with the magnitude of each [ComplexFloat] element in [a].
 *
 * The result element type is [Float] (the real-valued magnitude).
 */
@JvmName("absComplexFloat")
public fun <D : Dimension> abs(a: MultiArray<ComplexFloat, D>): NDArray<Float, D> {
    val ret = initMemoryView<Float>(a.size)
    var index = 0
    for (element in a) {
        ret[index++] = element.abs()
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/**
 * Returns a new array with the magnitude of each [ComplexDouble] element in [a].
 *
 * The result element type is [Double] (the real-valued magnitude).
 */
@JvmName("absComplexDouble")
public fun <D : Dimension> abs(a: MultiArray<ComplexDouble, D>): NDArray<Double, D> {
    val ret = initMemoryView<Double>(a.size)
    var index = 0
    for (element in a) {
        ret[index++] = element.abs()
    }
    return NDArray(ret, 0, a.shape.copyOf(), dim = a.dim)
}

/** Absolute value for [Byte] (not provided by kotlin.math). */
@Suppress( "nothing_to_inline")
private inline fun absByte(a: Byte): Byte = if (a < 0) (-a).toByte() else a

/** Absolute value for [Short] (not provided by kotlin.math). */
@Suppress( "nothing_to_inline")
private inline fun absShort(a: Short): Short = if (a < 0) (-a).toShort() else a
