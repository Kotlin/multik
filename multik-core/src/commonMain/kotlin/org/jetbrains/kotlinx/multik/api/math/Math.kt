/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Mathematical methods interface.
 */
public interface Math {

    /**
     * instance of [MathEx]
     */
    public val mathEx: MathEx

    /**
     * Returns flat index of maximum element in an ndarray.
     */
    public fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int

    /**
     * Returns an ndarray of indices of maximum elements in an ndarray [a] over a given [axis].
     */
    public fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>

    /**
     * Returns an ndarray of indices of maximum elements in a two-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1>

    /**
     * Returns an ndarray of indices of maximum elements in a three-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2>

    /**
     * Returns an ndarray of indices of maximum elements in a four-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3>

    /**
     * Returns an ndarray of indices of maximum elements in an n-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4>

    /**
     * Returns flat index of minimum element in an ndarray.
     */
    public fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int

    /**
     * Returns an ndarray of indices of minimum elements in an ndarray [a] over a given [axis].
     */
    public fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>

    /**
     * Returns an ndarray of indices of minimum elements in a two-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1>

    /**
     * Returns an ndarray of indices of minimum elements in a three-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2>

    /**
     * Returns an ndarray of indices of minimum elements in a four-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3>

    /**
     * Returns an ndarray of indices of minimum elements in an n-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4>

    /**
     * Returns maximum element of the given ndarray.
     */
    public fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T

    /**
     * Returns maximum of an ndarray [a] along a given [axis].
     */
    public fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /**
     * Returns maximum of a two-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /**
     * Returns maximum of a three-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /**
     * Returns maximum of a four-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /**
     * Returns maximum of an n-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4>

    /**
     * Returns minimum element of the given ndarray.
     */
    public fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T

    /**
     * Returns minimum of an ndarray [a] along a given [axis].
     */
    public fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /**
     * Returns minimum of a two-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /**
     * Returns minimum of a three-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /**
     * Returns minimum of a four-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /**
     * Returns minimum of an n-dimensional ndarray [a] along a given [axis].
     */
    public fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4>

    /**
     * Returns sum of all elements in the given ndarray.
     */
    public fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T

    /**
     * Returns an ndarray of sum all elements over a given [axis].
     */
    public fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

    /**
     * Returns an ndarray of sum all elements in a two-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1>

    /**
     * Returns an ndarray of sum all elements in a three-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2>

    /**
     * Returns an ndarray of sum all elements in a four-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3>

    /**
     * Returns an ndarray of sum all elements in a n-dimensional ndarray [a] over a given [axis].
     */
    public fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4>

    /**
     * Returns cumulative sum of all elements in the given ndarray.
     */
    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T>

    /**
     * Returns cumulative sum of all elements in the given ndarray along the given [axis].
     */
    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D>
}