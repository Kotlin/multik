/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

/**
 *  A generic ndarray. Methods in this interface support only read-only access to the ndarray.
 *
 *  @property data [MemoryView].
 *  @property offset Offset from the start of an ndarray's data.
 *  @property shape [IntArray] of an ndarray dimensions.
 *  @property strides [IntArray] indices to step in each dimension when iterating an ndarray.
 *  @property size number of elements in an ndarray.
 *  @property dtype [DataType] of an ndarray's data.
 *  @property dim [Dimension].
 *  @property base Base array if [data] is taken from other array. Otherwise - null.
 *  @property consistent indicates whether the array data is homogeneous.
 *  @property indices indices for a one-dimensional ndarray.
 *  @property multiIndices indices for a n-dimensional ndarray.
 */
public interface MultiArray<T, D : Dimension> {
    public val data: ImmutableMemoryView<T>
    public val offset: Int
    public val shape: IntArray
    public val strides: IntArray
    public val size: Int
    public val dtype: DataType
        get() = data.dtype
    public val dim: D
    public val base: MultiArray<T, out Dimension>?

    public val consistent: Boolean

    public val indices: IntRange
    public val multiIndices: MultiIndexProgression

    /**
     * Returns `true` if the array contains only one element, otherwise `false`.
     */
    public fun isScalar(): Boolean

    /**
     * Returns `true` if this ndarray is empty.
     */
    public fun isEmpty(): Boolean

    /**
     * Returns `true` if this ndarray is not empty.
     */
    public fun isNotEmpty(): Boolean

    /**
     * Returns new [MultiArray] which is a copy of the original ndarray.
     */
    public fun copy(): MultiArray<T, D>

    /**
     * Returns new [MultiArray] which is a deep copy of the original ndarray.
     */
    public fun deepCopy(): MultiArray<T, D>

    public operator fun iterator(): Iterator<T>

    /**
     * Returns new one-dimensional ndarray which is a copy of the original ndarray.
     */
    public fun flatten(): MultiArray<T, D1>


    // Reshape
    /**
     * Returns an ndarray with a new ([dim1]) shape without changing data.
     */
    public fun reshape(dim1: Int): MultiArray<T, D1>

    /**
     * Returns an ndarray with a new ([dim1], [dim2]) shape without changing data.
     */
    public fun reshape(dim1: Int, dim2: Int): MultiArray<T, D2>

    /**
     * Returns an ndarray with a new ([dim1], [dim2], [dim3]) shape without changing data.
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int): MultiArray<T, D3>

    /**
     * Returns an ndarray with a new ([dim1], [dim2], [dim3], [dim4]) shape without changing data.
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MultiArray<T, D4>

    /**
     * Returns an ndarray with a new ([dim1], [dim2], [dim3], [dim4], [dims]) shape without changing data.
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): MultiArray<T, DN>

    /**
     * Reverse or permute the [axes] of an array.
     */
    public fun transpose(vararg axes: Int): MultiArray<T, D>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns an ndarray with all axes removed equal to one.
     */
    public fun squeeze(vararg axes: Int): MultiArray<T, DN>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns a new ndarray with a dimension of size one inserted at the specified [axes].
     */
    public fun unsqueeze(vararg axes: Int): MultiArray<T, DN>

    /**
     * Concatenates this ndarray with [other].
     */
    public infix fun cat(other: MultiArray<T, D>): NDArray<T, D>

    /**
     * Concatenates this ndarray with [other] along the specified [axis].
     */
    public fun cat(other: MultiArray<T, D>, axis: Int = 0): NDArray<T, D>

    /**
     * Concatenates this ndarray with a list of [other] ndarrays.
     */
    public fun cat(other: List<MultiArray<T, D>>, axis: Int = 0): NDArray<T, D>
}

public fun <T, D : Dimension> MultiArray<T, D>.asDNArray(): NDArray<T, DN> {
    if (this is NDArray<T, D>)
        return this.asDNArray()
    else throw ClassCastException("Cannot cast MultiArray to NDArray of dimension n.")
}
