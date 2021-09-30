/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
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
     * Returns an ndarray with a new shape without changing data.
     */
    public fun reshape(dim1: Int): MultiArray<T, D1>

    public fun reshape(dim1: Int, dim2: Int): MultiArray<T, D2>

    public fun reshape(dim1: Int, dim2: Int, dim3: Int): MultiArray<T, D3>

    public fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MultiArray<T, D4>

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
    public fun cat(other: MultiArray<T, D>, axis: Int = 0): NDArray<T, D>
}

//___________________________________________________ReadableView_______________________________________________________

public class ReadableView<T>(private val base: MultiArray<T, DN>) /*: BaseNDArray by base */ {
    public operator fun get(vararg indices: Int): MultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.view(pos) }
    }
}

public fun <T, D : Dimension, M : Dimension> MultiArray<T, D>.view(index: Int, axis: Int = 0): MultiArray<T, M> {
    checkBounds(index in 0 until shape[axis], index, axis, axis)
    return NDArray(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), dimensionOf(this.dim.d - 1), base ?: this
    )
}

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

public val <T> MultiArray<T, DN>.V: ReadableView<T>
    get() = ReadableView(this)

//____________________________________________________Get_______________________________________________________________

@JvmName("get0")
public operator fun <T> MultiArray<T, D1>.get(index: Int): T {
    checkBounds(index in 0 until this.shape[0], index, 0, this.shape[0])
    return data[offset + strides.first() * index]
}

@JvmName("get1")
public operator fun <T> MultiArray<T, D2>.get(index: Int): MultiArray<T, D1> = view(index, 0)

@JvmName("get2")
public operator fun <T> MultiArray<T, D2>.get(ind1: Int, ind2: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    return data[offset + strides[0] * ind1 + strides[1] * ind2]
}

@JvmName("get3")
public operator fun <T> MultiArray<T, D3>.get(index: Int): MultiArray<T, D2> = view(index, 0)

@JvmName("get4")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: Int): MultiArray<T, D1> =
    view(ind1, ind2, 0, 1)

@JvmName("get5")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: Int, ind3: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    return data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3]
}

@JvmName("get6")
public operator fun <T> MultiArray<T, D4>.get(index: Int): MultiArray<T, D3> = view(index, 0)

@JvmName("get7")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int): MultiArray<T, D2> =
    view(ind1, ind2, 0, 1)

@JvmName("get8")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int): MultiArray<T, D1> =
    view(ind1, ind2, ind3, 0, 1, 2)

@JvmName("get9")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int, ind4: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    checkBounds(ind4 in 0 until this.shape[3], ind4, 3, this.shape[3])
    return data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4]
}

@JvmName("get10")
public operator fun <T> MultiArray<T, DN>.get(vararg index: Int): T = this[index]

@JvmName("get11")
public operator fun <T> MultiArray<T, DN>.get(index: IntArray): T {
    check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
    for (i in index.indices)
        checkBounds(index[i] in 0 until this.shape[i], index[i], i, this.shape[i])
    return data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }]
}

//_______________________________________________GetWithSlice___________________________________________________________

public fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(inSlice: ClosedRange<Int>, axis: Int = 0): NDArray<T, O> {
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
        shape[axis]
    }

    val sliceStrides = strides.clone().apply { this[axis] *= slice.step }
    val sliceShape = if (actualFrom > actualTo) {
        intArrayOf(0)
    } else {
        shape.clone().apply {
            this[axis] = (actualTo - actualFrom + slice.step - 1) / slice.step
        }
    }
    return NDArray(
        data,
        offset + actualFrom * strides[axis],
        sliceShape,
        sliceStrides,
        dimensionOf(sliceShape.size),
        base ?: this
    )
}


public fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(indexing: Map<Int, Indexing>): NDArray<T, O> {
    var newOffset = offset
    var newShape: IntArray = shape.clone()
    var newStrides: IntArray = strides.clone()
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

                val actualTo = if (index.start != -1) {
                    check(index.stop <= shape[ind.key]) { "slicing end index out of bounds: ${index.stop} > ${shape[ind.key]}" }
                    index.stop
                } else {
                    check(shape[ind.key] > index.start) { "slicing start index out of bounds: $actualFrom >= ${shape[ind.key]}" }
                    shape[ind.key]
                }

                newOffset += actualFrom * newStrides[ind.key]
                newShape[ind.key] = if (actualFrom > actualTo) 0 else (actualTo - actualFrom + index.step - 1) / index.step
                newStrides[ind.key] *= index.step
            }
        }
    }

    newShape = newShape.removeAll(removeAxes)
    newStrides = newStrides.removeAll(removeAxes)
    return NDArray(this.data, newOffset, newShape, newStrides, dimensionOf(newShape.size), base ?: this)
}

@JvmName("get12")
public operator fun <T> MultiArray<T, D1>.get(index: ClosedRange<Int>): MultiArray<T, D1> = slice(index)

@JvmName("get13")
public operator fun <T> MultiArray<T, D2>.get(index: ClosedRange<Int>): MultiArray<T, D2> = slice(index)

@JvmName("get14")
public operator fun <T> MultiArray<T, D2>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))

@JvmName("get15")
public operator fun <T> MultiArray<T, D2>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("get16")
public operator fun <T> MultiArray<T, D2>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("get17")
public operator fun <T> MultiArray<T, D3>.get(index: ClosedRange<Int>): MultiArray<T, D3> = slice(index)

@JvmName("get18")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))

@JvmName("get19")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("get20")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("get21")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("get22")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("get23")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("get24")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r))

@JvmName("get25")
public operator fun <T> MultiArray<T, D3>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("get26")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("get27")
public operator fun <T> MultiArray<T, D3>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("get28")
public operator fun <T> MultiArray<T, D4>.get(index: ClosedRange<Int>): MultiArray<T, D4> =
    slice(index)

@JvmName("get29")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice()))


@JvmName("get30")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice()))

@JvmName("get31")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r))

@JvmName("get32")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("get33")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("get34")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("get35")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r))

@JvmName("get36")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice()))

@JvmName("get37")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r))

@JvmName("get38")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice()))

@JvmName("get39")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>): MultiArray<T, D4> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("get39")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int, ind4: ClosedRange<Int>): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("get40")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>, ind4: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("get41")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int, ind4: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.r))

@JvmName("get42")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int, ind4: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r, 3 to ind4.r))

@JvmName("get43")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("get44")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Slice, ind4: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("get45")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int, ind4: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.r))

@JvmName("get46")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: Int, ind4: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("get47")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: Int, ind4: ClosedRange<Int>): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("get48")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>, ind4: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.r))

@JvmName("get49")
public operator fun <T> MultiArray<T, D4>.get(ind1: Int, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("get50")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: Int, ind3: ClosedRange<Int>, ind4: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.r, 2 to ind3.toSlice(), 3 to ind4.toSlice()))

@JvmName("get51")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: Int, ind4: ClosedRange<Int>): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.r, 3 to ind4.toSlice()))

@JvmName("get52")
public operator fun <T> MultiArray<T, D4>.get(ind1: ClosedRange<Int>, ind2: ClosedRange<Int>, ind3: ClosedRange<Int>, ind4: Int): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.toSlice(), 1 to ind2.toSlice(), 2 to ind3.toSlice(), 3 to ind4.r))

public fun <T> MultiArray<T, DN>.slice(map: Map<Int, Indexing>): MultiArray<T, DN> =
    slice<T, DN, DN>(map)

//________________________________________________asDimension___________________________________________________________

public fun <T, D : Dimension> MultiArray<T, D>.asDNArray(): NDArray<T, DN> {
    if (this is NDArray<T, D>)
        return this.asDNArray()
    else throw ClassCastException("Cannot cast MultiArray to NDArray of dimension n.")
}

@Suppress("NOTHING_TO_INLINE")
public inline fun checkBounds(value: Boolean, index: Int, axis: Int, size: Int) {
    if (!value) {
        throw IndexOutOfBoundsException("Index $index is out of bounds shape dimension $axis with size $size")
    }
}