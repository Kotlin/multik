package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Mathematical operations interface, accessible via `mk.math`.
 *
 * Provides aggregation functions ([argMax], [argMin], [max], [min], [sum], [cumSum])
 * and delegates transcendental functions (sin, cos, exp, log) to [MathEx] via [mathEx].
 *
 * Axis-based overloads (e.g., `argMax(a, axis)`) require explicit dimension type parameters
 * or use the dimension-specific variants ([argMaxD2], [argMaxD3], [argMaxD4], [argMaxDN]).
 *
 * @see [MathEx] for transcendental functions (sin, cos, exp, log).
 */
public interface Math {

    /**
     * The extended math operations provider for transcendental functions.
     */
    public val mathEx: MathEx

    // ---- argMax ----

    /**
     * Returns the flat index of the maximum element in the ndarray.
     *
     * @param a the input array.
     * @return the flat index of the maximum element.
     */
    public fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int

    /**
     * Returns indices of maximum elements along the given [axis].
     *
     * @param a the input array.
     * @param axis the axis to reduce along.
     * @return an [NDArray] of indices with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>

    /** Dimension-specific [argMax] along [axis] for 2D arrays. */
    public fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1>

    /** Dimension-specific [argMax] along [axis] for 3D arrays. */
    public fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2>

    /** Dimension-specific [argMax] along [axis] for 4D arrays. */
    public fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3>

    /** Dimension-specific [argMax] along [axis] for N-dimensional arrays. */
    public fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN>

    // ---- argMin ----

    /**
     * Returns the flat index of the minimum element in the ndarray.
     *
     * @param a the input array.
     * @return the flat index of the minimum element.
     */
    public fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int

    /**
     * Returns indices of minimum elements along the given [axis].
     *
     * @param a the input array.
     * @param axis the axis to reduce along.
     * @return an [NDArray] of indices with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>

    /** Dimension-specific [argMin] along [axis] for 2D arrays. */
    public fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1>

    /** Dimension-specific [argMin] along [axis] for 3D arrays. */
    public fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2>

    /** Dimension-specific [argMin] along [axis] for 4D arrays. */
    public fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3>

    /** Dimension-specific [argMin] along [axis] for N-dimensional arrays. */
    public fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN>

    // ---- max ----

    /**
     * Returns the maximum element in the ndarray.
     *
     * @param a the input array.
     * @return the maximum element.
     */
    public fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T

    /**
     * Returns the maximum elements along the given [axis].
     *
     * @param a the input array.
     * @param axis the axis to reduce along.
     * @return an [NDArray] of maximums with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /** Dimension-specific [max] along [axis] for 2D arrays. */
    public fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /** Dimension-specific [max] along [axis] for 3D arrays. */
    public fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /** Dimension-specific [max] along [axis] for 4D arrays. */
    public fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /** Dimension-specific [max] along [axis] for N-dimensional arrays. */
    public fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN>

    // ---- min ----

    /**
     * Returns the minimum element in the ndarray.
     *
     * @param a the input array.
     * @return the minimum element.
     */
    public fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T

    /**
     * Returns the minimum elements along the given [axis].
     *
     * @param a the input array.
     * @param axis the axis to reduce along.
     * @return an [NDArray] of minimums with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /** Dimension-specific [min] along [axis] for 2D arrays. */
    public fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /** Dimension-specific [min] along [axis] for 3D arrays. */
    public fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /** Dimension-specific [min] along [axis] for 4D arrays. */
    public fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /** Dimension-specific [min] along [axis] for N-dimensional arrays. */
    public fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN>

    // ---- sum ----

    /**
     * Returns the sum of all elements in the ndarray.
     *
     * @param a the input array.
     * @return the sum of all elements.
     */
    public fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T

    /**
     * Returns the element-wise sums along the given [axis].
     *
     * @param a the input array.
     * @param axis the axis to reduce along.
     * @return an [NDArray] of sums with rank reduced by one.
     */
    public fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /** Dimension-specific [sum] along [axis] for 2D arrays. */
    public fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /** Dimension-specific [sum] along [axis] for 3D arrays. */
    public fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /** Dimension-specific [sum] along [axis] for 4D arrays. */
    public fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /** Dimension-specific [sum] along [axis] for N-dimensional arrays. */
    public fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN>

    // ---- cumSum ----

    /**
     * Returns the cumulative sum of all elements as a flattened 1D array.
     *
     * @param a the input array.
     * @return a [D1Array] where each element is the running total.
     */
    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T>

    /**
     * Returns the cumulative sum along the given [axis], preserving the array shape.
     *
     * @param a the input array.
     * @param axis the axis to accumulate along.
     * @return an [NDArray] of the same shape with cumulative sums along [axis].
     */
    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D>
}
