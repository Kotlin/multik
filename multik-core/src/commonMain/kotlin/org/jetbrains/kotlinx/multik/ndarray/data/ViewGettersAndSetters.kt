package org.jetbrains.kotlinx.multik.ndarray.data

import kotlin.jvm.JvmName

@Suppress( "nothing_to_inline")
internal inline fun <T> MultiArray<T, D1>.unsafeIndex(index: Int): Int = offset + strides.first() * index

@Suppress( "nothing_to_inline")
internal inline fun <T> MultiArray<T, D2>.unsafeIndex(ind1: Int, ind2: Int): Int =
    offset + strides[0] * ind1 + strides[1] * ind2

@Suppress( "nothing_to_inline")
internal inline fun <T> MultiArray<T, D3>.unsafeIndex(ind1: Int, ind2: Int, ind3: Int): Int =
    offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3

@Suppress( "nothing_to_inline")
internal inline fun <T> MultiArray<T, D4>.unsafeIndex(ind1: Int, ind2: Int, ind3: Int, ind4: Int): Int =
    offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4

@Suppress( "nothing_to_inline")
internal inline fun <T> MultiArray<T, *>.unsafeIndex(indices: IntArray): Int =
    strides.foldIndexed(offset) { i, acc, stride -> acc + indices[i] * stride }

// =================================== ReadableView ===================================

/**
 * Provides chained read-only sub-view access for [DN]-dimensional arrays via the [V] property.
 *
 * ```
 * val a: NDArray<Int, DN> = ...
 * val subView = a.V[0, 1] // selects along leading axes
 * ```
 *
 * @param T the element type.
 * @see [V]
 */
public class ReadableView<T>(private val base: MultiArray<T, DN>) {
    /** Returns a sub-view by selecting a single index along each leading axis in order. */
    public operator fun get(vararg indices: Int): MultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.view(pos) }
    }
}

/**
 * Returns a read-only view by selecting a single [index] along the given [axis], reducing dimensionality by one.
 *
 * The returned view shares the underlying data buffer (no copy).
 *
 * @param index the position to select along [axis].
 * @param axis the axis to index into (default 0).
 * @return a view with one fewer dimension.
 * @throws IndexOutOfBoundsException if [index] is out of range for [axis].
 */
public fun <T, D : Dimension, M : Dimension> MultiArray<T, D>.view(index: Int, axis: Int = 0): MultiArray<T, M> {
    checkBounds(index in 0 until shape[axis], index, axis, axis)
    return NDArray(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), dimensionOf(this.dim.d - 1), base ?: this
    )
}

/**
 * Returns a read-only view by selecting elements at [indices] along multiple [axes],
 * reducing dimensionality by `axes.size`.
 *
 * @param indices positions to select, one per axis in [axes].
 * @param axes the axes to index into.
 * @return a view with `axes.size` fewer dimensions.
 * @throws IndexOutOfBoundsException if any index is out of range.
 */
@Suppress("DuplicatedCode")
public fun <T, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    indices: IntArray, axes: IntArray
): MultiArray<T, M> {
    for ((ind, axis) in indices.zip(axes))
        checkBounds(ind in 0 until this.shape[axis], ind, axis, this.shape[axis])
    val newShape = shape.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    val newStrides = strides.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    var newOffset = offset
    for (i in axes.indices)
        newOffset += strides[axes[i]] * indices[i]
    return NDArray(data, newOffset, newShape, newStrides, dimensionOf(this.dim.d - axes.size), base ?: this)
}

@JvmName("viewD2")
public fun <T> MultiArray<T, D2>.view(index: Int, axis: Int = 0): MultiArray<T, D1> =
    view<T, D2, D1>(index, axis)

@JvmName("viewD3")
public fun <T> MultiArray<T, D3>.view(index: Int, axis: Int = 0): MultiArray<T, D2> =
    view<T, D3, D2>(index, axis)

@JvmName("viewD3toD1")
public fun <T> MultiArray<T, D3>.view(ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1): MultiArray<T, D1> =
    view(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4")
public fun <T> MultiArray<T, D4>.view(index: Int, axis: Int = 0): MultiArray<T, D3> =
    view<T, D4, D3>(index, axis)

@JvmName("viewD4toD2")
public fun <T> MultiArray<T, D4>.view(ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1): MultiArray<T, D2> =
    view(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4toD1")
public fun <T> MultiArray<T, D4>.view(
    ind1: Int, ind2: Int, ind3: Int, axis1: Int = 0, axis2: Int = 1, axis3: Int = 2
): MultiArray<T, D1> = view(intArrayOf(ind1, ind2, ind3), intArrayOf(axis1, axis2, axis3))

@JvmName("viewDN")
public fun <T> MultiArray<T, DN>.view(index: Int, axis: Int = 0): MultiArray<T, DN> =
    view<T, DN, DN>(index, axis)

@JvmName("viewDN")
public fun <T> MultiArray<T, DN>.view(index: IntArray, axes: IntArray): MultiArray<T, DN> =
    view<T, DN, DN>(index, axes)

/**
 * Read-only chained view accessor for [DN]-dimensional arrays.
 *
 * ```
 * val row = ndarray.V[0]  // first sub-array along axis 0
 * ```
 *
 * @see [ReadableView]
 */
public val <T> MultiArray<T, DN>.V: ReadableView<T>
    get() = ReadableView(this)

// =================================== View Get (sub-array by index) ===================================
// These operators select a sub-array along axis 0, returning a view with one fewer dimension.

/** Returns a 1D view of row [index] from this 2D array. */
@JvmName("getView1")
public operator fun <T> MultiArray<T, D2>.get(index: Int): MultiArray<T, D1> = view(index, 0)

/** Returns a 2D view at [index] along axis 0 from this 3D array. */
@JvmName("getView2")
public operator fun <T> MultiArray<T, D3>.get(index: Int): MultiArray<T, D2> = view(index, 0)

/** Returns a 1D view at ([ind1], [ind2]) along axes 0 and 1 from this 3D array. */
@JvmName("getView3")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: Int): MultiArray<T, D1> =
    view(ind1, ind2, 0, 1)

/** Returns a 3D view at [index] along axis 0 from this 4D array. */
@JvmName("getView4")
public operator fun <T> MultiArray<T, D4>.get(index: Int): MultiArray<T, D3> = view(index, 0)

@JvmName("getView5")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int): MultiArray<T, D2> =
    view(ind1, ind2, 0, 1)

@JvmName("getView6")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int): MultiArray<T, D1> =
    view(ind1, ind2, ind3, 0, 1, 2)

// =================================== Slicing ===================================
// Slice functions return views into the original data buffer (no copy).

/**
 * Returns a view of this array sliced along a single [axis] using the given range.
 *
 * The returned view shares the underlying data. Use `-1` as start/stop sentinel values
 * (via [sl.first]/[sl.last]) to indicate axis boundaries.
 *
 * @param inSlice the range to slice (supports [Slice], [IntRange]).
 * @param axis the axis to slice along (default 0).
 * @return a view with the sliced axis.
 * @throws IllegalArgumentException if [axis] is out of bounds.
 * @throws IllegalStateException if start or stop indices are out of range.
 */
public fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(
    inSlice: ClosedRange<Int>,
    axis: Int = 0
): NDArray<T, O> {
    require(axis in 0 until this.dim.d) { "axis out of bounds: $axis" }

    val slice = inSlice.toSlice()

    val actualFrom = if (slice.start != -1) {
        check(slice.start > -1) { "slicing start index must be positive, but was ${slice.start}" }
        slice.start
    } else {
        0
    }

    val actualTo = if (slice.stop != -1) {
        check(slice.stop <= shape[axis]) { "slicing end index out of bounds: ${slice.stop} > ${shape[axis]}" }
        slice.stop
    } else {
        check(shape[axis] > actualFrom) { "slicing start index out of bounds: $actualFrom >= ${shape[axis]}" }
        shape[axis] - 1
    }

    val sliceStrides = strides.copyOf().apply { this[axis] *= slice.step }
    val sliceShape = if (actualFrom > actualTo) {
        intArrayOf(0)
    } else {
        shape.copyOf().apply {
            this[axis] = (actualTo - actualFrom + slice.step) / slice.step
        }
    }
    return NDArray(
        data, offset + actualFrom * strides[axis],
        sliceShape, sliceStrides, dimensionOf(sliceShape.size),
        base ?: this
    )
}


/**
 * Returns a view of this array sliced along multiple axes using the given [indexing] map.
 *
 * Each entry maps an axis index to either a [Slice] (range selection) or an [RInt] (single-index selection
 * that removes the axis). The returned view shares the underlying data.
 *
 * @param indexing map from axis index to [Slice] or [RInt].
 * @return a view with potentially reduced dimensionality.
 * @throws IllegalArgumentException if any axis is out of bounds or any index is out of range.
 */
public fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(indexing: Map<Int, Indexing>): NDArray<T, O> {
    var newOffset = offset
    var newShape: IntArray = shape.copyOf()
    var newStrides: IntArray = strides.copyOf()
    val removeAxes = mutableListOf<Int>()
    for (ind in indexing) {
        require(ind.key in 0 until this.dim.d) { "axis out of bounds: ${ind.key}" }
        when (ind.value) {
            is RInt -> {
                val index = (ind.value as RInt).data
                require(index in 0 until shape[ind.key]) { "Index $index out of bounds at [0, ${shape[ind.key] - 1}]" }

                newOffset += newStrides[ind.key] * index
                removeAxes.add(ind.key)
            }

            is Slice -> {
                val index = ind.value as Slice

                val actualFrom = if (index.start != -1) {
                    check(index.start > -1) { "slicing start index must be positive, but was ${index.start}" }
                    index.start
                } else {
                    0
                }

                val actualTo = if (index.stop != -1) {
                    check(index.stop <= shape[ind.key]) { "slicing end index out of bounds: ${index.stop} > ${shape[ind.key]}" }
                    index.stop
                } else {
                    check(shape[ind.key] > index.start) { "slicing start index out of bounds: $actualFrom >= ${shape[ind.key]}" }
                    shape[ind.key] - 1
                }

                newOffset += actualFrom * newStrides[ind.key]
                newShape[ind.key] = if (actualFrom > actualTo) 0 else (actualTo - actualFrom + index.step) / index.step
                newStrides[ind.key] *= index.step
            }
        }
    }

    newShape = newShape.removeAll(removeAxes)
    newStrides = newStrides.removeAll(removeAxes)
    return NDArray(this.data, newOffset, newShape, newStrides, dimensionOf(newShape.size), base ?: this)
}

@JvmName("getView7")
public operator fun <T> MultiArray<T, D1>.get(index: ClosedRange<Int>): MultiArray<T, D1> = slice(index)

@JvmName("getView8")
public operator fun <T> MultiArray<T, D2>.get(index: ClosedRange<Int>): MultiArray<T, D2> = slice(index)

@JvmName("getView9")
public operator fun <T> MultiArray<T, D2>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))

@JvmName("getView10")
public operator fun <T> MultiArray<T, D2>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("getView11")
public operator fun <T> MultiArray<T, D2>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("getView12")
public operator fun <T> MultiArray<T, D3>.get(index: ClosedRange<Int>): MultiArray<T, D3> = slice(index)

@JvmName("getView13")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))

@JvmName("getView14")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("getView15")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("getView16")
public operator fun <T> MultiArray<T, D3>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("getView17")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("getView18")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("getView19")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r))

@JvmName("getView20")
public operator fun <T> MultiArray<T, D3>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("getView21")
public operator fun <T> MultiArray<T, D3>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("getView22")
public operator fun <T> MultiArray<T, D3>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("getView23")
public operator fun <T> MultiArray<T, D4>.get(index: ClosedRange<Int>): MultiArray<T, D4> =
    slice(index)

@JvmName("getView24")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))


@JvmName("getView25")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("getView26")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("getView27")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>
): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("getView28")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("getView29")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("getView30")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r))

@JvmName("getView31")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("getView32")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("getView33")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("getView34")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>,
    ind2: ClosedRange<Int>,
    ind3: ClosedRange<Int>,
    ind4: ClosedRange<Int>
): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("getView35")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: Int, ind3: Int, ind4: ClosedRange<Int>
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("getView36")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: Int, ind3: ClosedRange<Int>, ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("getView37")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: Int, ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.r))

@JvmName("getView38")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: Int, ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r, 3 to ind4.r))

@JvmName("getView39")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: Int, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("getView40")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("getView41")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int, ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.r))

@JvmName("getView42")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: Int, ind4: ClosedRange<Int>
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("getView43")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: Int, ind4: ClosedRange<Int>
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("getView44")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>, ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("getView45")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("getView46")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("getView47")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int, ind4: ClosedRange<Int>
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("getView48")
public operator fun <T> MultiArray<T, D4>.get(
    ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: Int
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.r))

/** Slices a [DN]-dimensional array along multiple axes using the given [map] of axis-to-[Indexing]. */
public fun <T> MultiArray<T, DN>.slice(map: Map<Int, Indexing>): MultiArray<T, DN> =
    slice<T, DN, DN>(map)

// =================================== WritableView ===================================

/**
 * Provides chained mutable sub-view access for [DN]-dimensional arrays via the [W] property.
 *
 * ```
 * val a: NDArray<Int, DN> = ...
 * a.W[0, 1][2] = 42  // write through the view
 * ```
 *
 * @param T the element type.
 * @see [W]
 */
public class WritableView<T>(private val base: MutableMultiArray<T, DN>) {
    /** Returns a mutable sub-view by selecting a single index along each leading axis in order. */
    public operator fun get(vararg indices: Int): MutableMultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.mutableView(pos) }
    }

    public companion object
}

/**
 * Returns a mutable view by selecting a single [index] along the given [axis].
 *
 * @see [view]
 * @see [mutableView]
 */
public inline fun <T, D : Dimension, reified M : Dimension> MultiArray<T, D>.writableView(
    index: Int, axis: Int = 0
): MutableMultiArray<T, M> {
    checkBounds(index in 0 until shape[axis], index, axis, axis)
    return NDArray(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), dimensionClassOf(this.dim.d - 1), base ?: this
    )
}

/**
 * Returns a mutable view by selecting [indices] along multiple [axes].
 *
 * @see [view]
 * @see [mutableView]
 */
public inline fun <T, D : Dimension, reified M : Dimension> MultiArray<T, D>.writableView(
    indices: IntArray, axes: IntArray
): MutableMultiArray<T, M> {
    for ((ind, axis) in indices.zip(axes))
        checkBounds(ind in 0 until this.shape[axis], ind, axis, this.shape[axis])
    val newShape = shape.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    val newStrides = strides.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    var newOffset = offset
    for (i in axes.indices)
        newOffset += strides[axes[i]] * indices[i]
    return NDArray(data, newOffset, newShape, newStrides, dimensionOf(this.dim.d - axes.size), base ?: this)
}

/** Returns a mutable view by selecting a single [index] along the given [axis]. Alias for [writableView]. */
public inline fun <T, D : Dimension, reified M : Dimension> MutableMultiArray<T, D>.mutableView(
    index: Int, axis: Int = 0
): MutableMultiArray<T, M> = this.writableView(index, axis)

/** Returns a mutable view by selecting [index] along multiple [axes]. Alias for [writableView]. */
public inline fun <T, D : Dimension, reified M : Dimension> MutableMultiArray<T, D>.mutableView(
    index: IntArray, axes: IntArray
): MutableMultiArray<T, M> = this.writableView(index, axes)

@JvmName("mutableViewD2")
public fun <T> MutableMultiArray<T, D2>.mutableView(index: Int, axis: Int = 0): MutableMultiArray<T, D1> =
    mutableView<T, D2, D1>(index, axis)

@JvmName("mutableViewD3")
public fun <T> MutableMultiArray<T, D3>.mutableView(index: Int, axis: Int = 0): MutableMultiArray<T, D2> =
    mutableView<T, D3, D2>(index, axis)

@JvmName("mutableViewD3toD1")
public fun <T> MutableMultiArray<T, D3>.mutableView(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MutableMultiArray<T, D1> = mutableView(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("mutableViewD4")
public fun <T> MutableMultiArray<T, D4>.mutableView(index: Int, axis: Int = 0): MutableMultiArray<T, D3> =
    mutableView<T, D4, D3>(index, axis)

@JvmName("mutableViewD4toD2")
public fun <T> MutableMultiArray<T, D4>.mutableView(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MutableMultiArray<T, D2> = mutableView(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("mutableViewD4toD1")
public fun <T> MutableMultiArray<T, D4>.mutableView(
    ind1: Int, ind2: Int, ind3: Int, axis1: Int = 0, axis2: Int = 1, axis3: Int = 2
): MutableMultiArray<T, D1> = mutableView(intArrayOf(ind1, ind2, ind3), intArrayOf(axis1, axis2, axis3))

@JvmName("mutableViewDN")
public fun <T> MutableMultiArray<T, DN>.mutableView(index: Int, axis: Int = 0): MutableMultiArray<T, DN> =
    mutableView<T, DN, DN>(index, axis)

@JvmName("mutableViewDN")
public fun <T> MutableMultiArray<T, DN>.mutableView(index: IntArray, axes: IntArray): MutableMultiArray<T, DN> =
    mutableView<T, DN, DN>(index, axes)

/**
 * Mutable chained view accessor for [DN]-dimensional arrays.
 *
 * ```
 * ndarray.W[0, 1][2] = 42
 * ```
 *
 * @see [WritableView]
 */
public val <T> MutableMultiArray<T, DN>.W: WritableView<T>
    get() = WritableView(this)

// =================================== Deprecated WritableView.Companion Get ===================================
// These are deprecated in favor of mutableView() calls. Retained for binary compatibility.

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(index, 0)"), level = DeprecationLevel.WARNING
)
@JvmName("get1")
public operator fun <T> MutableMultiArray<T, D2>.get(
    write: WritableView.Companion, index: Int
): MutableMultiArray<T, D1> = mutableView(index, 0)

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(index, 0)"), level = DeprecationLevel.WARNING
)
@JvmName("get2")
public operator fun <T> MutableMultiArray<T, D3>.get(
    write: WritableView.Companion, index: Int
): MutableMultiArray<T, D2> = mutableView(index, 0)

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(ind1, ind2, 0, 1)"), level = DeprecationLevel.WARNING
)
@JvmName("get3")
public operator fun <T> MutableMultiArray<T, D3>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int
): MultiArray<T, D1> = mutableView(ind1, ind2, 0, 1)

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(index, 0)"), level = DeprecationLevel.WARNING
)
@JvmName("get4")
public operator fun <T> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion, index: Int
): MutableMultiArray<T, D3> = mutableView(index, 0)

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(ind1, ind2, 0, 1)"), level = DeprecationLevel.WARNING
)
@JvmName("get5")
public operator fun <T> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int
): MultiArray<T, D2> = mutableView(ind1, ind2, 0, 1)

@Deprecated(
    """
    This function returns a mutable array, which might lead to unintended side effects due to possible external modifications.
    If you need to modify the array, consider creating a copy or use alternative methods that ensure data integrity.
""", replaceWith = ReplaceWith("mutableView(ind1, ind2, ind3, 0, 1, 2)"), level = DeprecationLevel.WARNING
)
@JvmName("get6")
public operator fun <T> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int, ind3: Int
): MutableMultiArray<T, D1> = mutableView(ind1, ind2, ind3, 0, 1, 2)

// =================================== Sub-array Set (assign view slices) ===================================
// These operators assign an entire sub-array (e.g. a row) by copying element-wise into the mutable view.

/** Assigns [value] into the 1D sub-array at row [index] of this 2D array. */
@JvmName("set1")
public operator fun <T> MutableMultiArray<T, D2>.set(index: Int, value: MultiArray<T, D1>) {
    val ret = this.mutableView(index, 0)
    requireArraySizes(ret.size, value.size)
    for (i in ret.indices)
        ret[i] = value[i]
}

/** Assigns [value] into the 2D sub-array at [index] along axis 0 of this 3D array. */
@JvmName("set2")
public operator fun <T> MutableMultiArray<T, D3>.set(index: Int, value: MultiArray<T, D2>) {
    val ret = this.mutableView(index, 0)
    requireArraySizes(ret.size, value.size)
    for ((i, j) in ret.multiIndices)
        ret[i, j] = value[i, j]
}

/** Assigns [value] into the 1D sub-array at ([ind1], [ind2]) of this 3D array. */
@JvmName("set3")
public operator fun <T> MutableMultiArray<T, D3>.set(ind1: Int, ind2: Int, value: MultiArray<T, D1>) {
    val ret = this.mutableView(ind1, ind2, 0, 1)
    requireArraySizes(ret.size, value.size)
    for (i in ret.indices)
        ret[i] = value[i]
}

/** Assigns [value] into the 3D sub-array at [index] along axis 0 of this 4D array. */
@JvmName("set4")
public operator fun <T> MutableMultiArray<T, D4>.set(index: Int, value: MultiArray<T, D3>) {
    val ret = this.mutableView(index, 0)
    requireArraySizes(ret.size, value.size)
    for ((i, j, k) in ret.multiIndices)
        ret[i, j, k] = value[i, j, k]
}

/** Assigns [value] into the 2D sub-array at ([ind1], [ind2]) of this 4D array. */
@JvmName("set5")
public operator fun <T> MutableMultiArray<T, D4>.set(ind1: Int, ind2: Int, value: MultiArray<T, D2>) {
    val ret = this.mutableView(ind1, ind2, 0, 1)
    requireArraySizes(ret.size, value.size)
    for ((i, j) in ret.multiIndices)
        ret[i, j] = value[i, j]
}

/** Assigns [value] into the 1D sub-array at ([ind1], [ind2], [ind3]) of this 4D array. */
@JvmName("set6")
public operator fun <T> MutableMultiArray<T, D4>.set(ind1: Int, ind2: Int, ind3: Int, value: MultiArray<T, D1>) {
    val ret = this.mutableView(ind1, ind2, ind3, 0, 1, 2)
    requireArraySizes(ret.size, value.size)
    for (i in ret.indices)
        ret[i] = value[i]
}
