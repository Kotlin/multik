package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Read-only interface for multidimensional arrays.
 *
 * Provides access to array metadata ([shape], [strides], [offset]) and structural operations
 * (reshape, transpose, squeeze) without mutation. The concrete implementation is [NDArray].
 *
 * @property data the flat memory buffer storing the array elements.
 * @property offset starting position in [data] for this array or view.
 * @property shape sizes of each dimension, e.g. `intArrayOf(3, 4)` for a 3x4 matrix.
 * @property strides number of elements to skip in [data] to advance one position along each dimension.
 * @property size total number of elements (product of [shape]).
 * @property dtype the [DataType] of the elements.
 * @property dim the [Dimension] type token (e.g. [D1], [D2], [DN]).
 * @property base the parent array when this is a view, or `null` for root arrays.
 * @property consistent `true` when the data is contiguous (`offset == 0` and `size == data.size`).
 * @property indices element index range for 1D arrays.
 * @property multiIndices [MultiIndexProgression] over all valid index tuples.
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
     * Returns a copy of this array that shares no data with the original.
     *
     * The underlying [MemoryView] buffer is copied, but the copy retains the same [offset] and [strides]
     * layout. Use [deepCopy] to produce a compact contiguous copy.
     *
     * @see [deepCopy]
     */
    public fun copy(): MultiArray<T, D>

    /**
     * Returns a contiguous deep copy of this array.
     *
     * Unlike [copy], this always allocates a new buffer containing only the meaningful elements
     * with `offset == 0` and default strides. Useful after slicing or transposing to reclaim memory.
     *
     * @see [copy]
     */
    public fun deepCopy(): MultiArray<T, D>

    /**
     * Returns an iterator over the elements of this ndarray in row-major order.
     *
     * For non-[consistent] arrays (views), iteration respects [offset] and [strides].
     */
    public operator fun iterator(): Iterator<T>

    /**
     * Returns a new 1D array containing all elements of this array in row-major order.
     *
     * Always allocates a new buffer (copy semantics).
     *
     * @return a [D1Array] of size [size].
     */
    public fun flatten(): MultiArray<T, D1>


    // Reshape

    /**
     * Returns a view of this array with the given 1D shape.
     *
     * The new shape must have the same total [size]. Returns a view when the array is [consistent];
     * otherwise performs a deep copy first.
     *
     * @param dim1 size of the single dimension (must equal [size]).
     * @return a 1D view or copy with shape `[dim1]`.
     * @throws IllegalArgumentException if [dim1] is not positive or does not equal [size].
     */
    public fun reshape(dim1: Int): MultiArray<T, D1>

    /**
     * Returns a view of this array with shape ([dim1], [dim2]).
     *
     * @param dim1 size of the first dimension.
     * @param dim2 size of the second dimension.
     * @return a 2D view or copy with shape `[dim1, dim2]`.
     * @throws IllegalArgumentException if the product of dimensions does not equal [size].
     * @see [reshape]
     */
    public fun reshape(dim1: Int, dim2: Int): MultiArray<T, D2>

    /**
     * Returns a view of this array with shape ([dim1], [dim2], [dim3]).
     *
     * @param dim1 size of the first dimension.
     * @param dim2 size of the second dimension.
     * @param dim3 size of the third dimension.
     * @return a 3D view or copy with shape `[dim1, dim2, dim3]`.
     * @throws IllegalArgumentException if the product of dimensions does not equal [size].
     * @see [reshape]
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int): MultiArray<T, D3>

    /**
     * Returns a view of this array with shape ([dim1], [dim2], [dim3], [dim4]).
     *
     * @param dim1 size of the first dimension.
     * @param dim2 size of the second dimension.
     * @param dim3 size of the third dimension.
     * @param dim4 size of the fourth dimension.
     * @return a 4D view or copy with shape `[dim1, dim2, dim3, dim4]`.
     * @throws IllegalArgumentException if the product of dimensions does not equal [size].
     * @see [reshape]
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MultiArray<T, D4>

    /**
     * Returns a view of this array with shape ([dim1], [dim2], [dim3], [dim4], [dims]).
     *
     * @param dim1 size of the first dimension.
     * @param dim2 size of the second dimension.
     * @param dim3 size of the third dimension.
     * @param dim4 size of the fourth dimension.
     * @param dims sizes of remaining dimensions.
     * @return an N-dimensional view or copy.
     * @throws IllegalArgumentException if the product of all dimensions does not equal [size].
     * @see [reshape]
     */
    public fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): MultiArray<T, DN>

    /**
     * Returns a view with permuted axes. Does not copy data.
     *
     * When called with no arguments, reverses all axes (equivalent to a full transpose).
     * When called with [axes], reorders dimensions according to the given permutation.
     *
     * @param axes optional permutation of `0 until dim.d`. Must be a complete permutation with unique values.
     * @return a view sharing the same data with reordered [shape] and [strides].
     * @throws IllegalArgumentException if [axes] is non-empty but incomplete, out of range, or contains duplicates.
     * @see [reshape]
     */
    public fun transpose(vararg axes: Int): MultiArray<T, D>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns a view with size-one dimensions removed.
     *
     * When called with no arguments, removes all axes whose size is 1.
     * When called with specific [axes], removes only those axes (they must have size 1).
     *
     * @param axes optional indices of axes to squeeze. Each must have `shape[axis] == 1`.
     * @return a view with reduced dimensionality.
     * @throws IllegalArgumentException if any specified axis does not have size 1.
     * @see [unsqueeze]
     */
    public fun squeeze(vararg axes: Int): MultiArray<T, DN>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns a view with a size-one dimension inserted at each of the specified [axes].
     *
     * @param axes positions at which to insert new dimensions of size 1.
     * @return a view with increased dimensionality.
     * @see [squeeze]
     */
    public fun unsqueeze(vararg axes: Int): MultiArray<T, DN>

    /**
     * Concatenates this array with [other] along axis 0, returning a new array.
     *
     * @param other the array to concatenate with this one.
     * @return a new [NDArray] containing elements from both arrays.
     * @throws IllegalArgumentException if shapes are incompatible on non-concatenation axes.
     */
    public infix fun cat(other: MultiArray<T, D>): NDArray<T, D>

    /**
     * Concatenates this array with [other] along the given [axis], returning a new array.
     *
     * @param other the array to concatenate with this one.
     * @param axis the axis along which to concatenate (default 0). Supports negative indexing.
     * @return a new [NDArray] with increased size along [axis].
     * @throws IllegalArgumentException if [axis] is out of bounds or shapes are incompatible.
     */
    public fun cat(other: MultiArray<T, D>, axis: Int = 0): NDArray<T, D>

    /**
     * Concatenates this array with a list of [other] arrays along the given [axis].
     *
     * @param other arrays to concatenate after this one, in order.
     * @param axis the axis along which to concatenate (default 0).
     * @return a new [NDArray] with combined size along [axis].
     * @throws IllegalArgumentException if [axis] is out of bounds or shapes are incompatible.
     */
    public fun cat(other: List<MultiArray<T, D>>, axis: Int = 0): NDArray<T, D>
}

/**
 * Casts this [MultiArray] to an [NDArray] with [DN] dimension type.
 *
 * @throws ClassCastException if this is not an [NDArray] instance.
 * @see [NDArray.asDNArray]
 */
public fun <T, D : Dimension> MultiArray<T, D>.asDNArray(): NDArray<T, DN> {
    if (this is NDArray<T, D>)
        return this.asDNArray()
    else throw ClassCastException("Cannot cast MultiArray to NDArray of dimension n.")
}
