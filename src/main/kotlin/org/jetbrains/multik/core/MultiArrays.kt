package org.jetbrains.multik.core

interface MultiArray<out T : Number, out D : Dimension> {
    val indices: IntRange
    val multiIndices: MultiIndexProgression

    public fun isEmpty(): Boolean
    public fun isNotEmpty(): Boolean

    public operator fun iterator(): Iterator<T>
}


//_________________________________________________Property_____________________________________________________________

internal val <T : Number, D : Dimension> MultiArray<T, D>.data: MemoryView<T>
    get() = (this as Ndarray<T, D>).data

internal val <T : Number, D : Dimension> MultiArray<T, D>.offset: Int
    get() = (this as Ndarray<T, D>).offset

@PublishedApi
internal val <T : Number, D : Dimension> MultiArray<T, D>.shape: IntArray
    get() = (this as Ndarray<T, D>).shape

internal val <T : Number, D : Dimension> MultiArray<T, D>.strides: IntArray
    get() = (this as Ndarray<T, D>).strides

@PublishedApi
internal val <T : Number, D : Dimension> MultiArray<T, D>.size: Int
    get() = (this as Ndarray<T, D>).size

@PublishedApi
internal val <T : Number, D : Dimension> MultiArray<T, D>.dtype: DataType
    get() = (this as Ndarray<T, D>).dtype

@PublishedApi
internal val <T : Number, D : Dimension> MultiArray<T, D>.dim: D
    get() = (this as Ndarray<T, D>).dim


//___________________________________________________ReadableView_______________________________________________________

public class ReadableView<T : Number>(private val base: MultiArray<T, DN>) /*: BaseNdarray by base */ {
    operator fun get(vararg indices: Int): MultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.view(pos) }
    }
}

public fun <T : Number, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    index: Int, axis: Int = 0
): MultiArray<T, M> {
    return Ndarray<T, M>(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), this.dtype, dimensionOf(this.dim.d - 1)
    )
}

public fun <T : Number, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    index: IntArray, axes: IntArray
): MultiArray<T, M> {
    val newShape = shape.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    val newStrides = strides.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    var newOffset = offset
    for (i in axes.indices)
        newOffset += strides[axes[i]] * index[i]
    return Ndarray<T, M>(data, newOffset, newShape, newStrides, this.dtype, dimensionOf(this.dim.d - axes.size))
}

@JvmName("viewD2")
public fun <T : Number> MultiArray<T, D2>.view(index: Int, axis: Int = 0): MultiArray<T, D1> =
    view<T, D2, D1>(index, axis)

@JvmName("viewD3")
public fun <T : Number> MultiArray<T, D3>.view(index: Int, axis: Int = 0): MultiArray<T, D2> =
    view<T, D3, D2>(index, axis)

@JvmName("viewD3toD1")
public fun <T : Number> MultiArray<T, D3>.view(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MultiArray<T, D1> = view<T, D3, D1>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4")
public fun <T : Number> MultiArray<T, D4>.view(index: Int, axis: Int = 0): MultiArray<T, D3> =
    view<T, D4, D3>(index, axis)

@JvmName("viewD4toD2")
public fun <T : Number> MultiArray<T, D4>.view(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MultiArray<T, D2> = view<T, D4, D2>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4toD1")
public fun <T : Number> MultiArray<T, D4>.view(
    ind1: Int, ind2: Int, ind3: Int, axis1: Int = 0, axis2: Int = 1, axis3: Int = 2
): MultiArray<T, D1> = view<T, D4, D1>(intArrayOf(ind1, ind2, ind3), intArrayOf(axis1, axis2, axis3))

@JvmName("viewDN")
public fun <T : Number> MultiArray<T, DN>.view(index: Int, axis: Int = 0): MultiArray<T, DN> =
    view<T, DN, DN>(index, axis)

@JvmName("viewDN")
public fun <T : Number> MultiArray<T, DN>.view(index: IntArray, axes: IntArray): MultiArray<T, DN> =
    view<T, DN, DN>(index, axes)

public val <T : Number> MultiArray<T, DN>.V: ReadableView<T>
    get() = ReadableView(this)

//____________________________________________________Get_______________________________________________________________

@JvmName("get0")
operator fun <T : Number> MultiArray<T, D1>.get(index: Int): T = data[offset + strides.first() * index]

@JvmName("get1")
operator fun <T : Number> MultiArray<T, D2>.get(index: Int): MultiArray<T, D1> = view(index, 0)

@JvmName("get2")
operator fun <T : Number> MultiArray<T, D2>.get(ind1: Int, ind2: Int): T =
    data[offset + strides[0] * ind1 + strides[1] * ind2]

@JvmName("get3")
operator fun <T : Number> MultiArray<T, D3>.get(index: Int): MultiArray<T, D2> = view(index, 0)

@JvmName("get4")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Int, ind2: Int): MultiArray<T, D1> =
    view(ind1, ind2, 0, 1)

@JvmName("get5")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Int, ind2: Int, ind3: Int): T =
    data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3]

@JvmName("get6")
operator fun <T : Number> MultiArray<T, D4>.get(index: Int): MultiArray<T, D3> = view(index, 0)

@JvmName("get7")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Int): MultiArray<T, D2> =
    view(ind1, ind2, 0, 1)

@JvmName("get8")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int): MultiArray<T, D1> =
    view(ind1, ind2, ind3, 0, 1, 2)

@JvmName("get9")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
    data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4]

@JvmName("get10")
operator fun <T : Number> MultiArray<T, DN>.get(vararg index: Int): T = this[index]

@JvmName("get11")
operator fun <T : Number> MultiArray<T, DN>.get(index: IntArray): T {
    check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
    return data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }]
}


//__________________________________________________Reshape_____________________________________________________________

public fun <T : Number, D : Dimension> MultiArray<T, D>.reshape(dim1: Int): MultiArray<T, D1> {
    if (this is Ndarray<T, D>)
        return this.reshape(dim1)
    else throw Exception("Cannot reshape.")
}

public fun <T : Number, D : Dimension> MultiArray<T, D>.reshape(dim1: Int, dim2: Int): MultiArray<T, D2> {
    if (this is Ndarray<T, D>)
        return this.reshape(dim1, dim2)
    else throw Exception("Cannot reshape.")
}

public fun <T : Number, D : Dimension> MultiArray<T, D>.reshape(dim1: Int, dim2: Int, dim3: Int): MultiArray<T, D3> {
    if (this is Ndarray<T, D>)
        return this.reshape(dim1, dim2, dim3)
    else throw Exception("Cannot reshape.")
}

public fun <T : Number, D : Dimension> MultiArray<T, D>.reshape(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int
): MultiArray<T, D4> {
    if (this is Ndarray<T, D>)
        return this.reshape(dim1, dim2, dim3, dim4)
    else throw Exception("Cannot reshape.")
}

public fun <T : Number, D : Dimension> MultiArray<T, D>.reshape(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): MultiArray<T, DN> {
    if (this is Ndarray<T, D>)
        return this.reshape(dim1, dim2, dim3, dim4, *dims)
    else throw Exception("Cannot reshape.")
}

//________________________________________________asDimension___________________________________________________________

public fun <T : Number, D : Dimension> MultiArray<T, D>.asDNArray(): Ndarray<T, DN> {
    if (this is Ndarray<T, D>)
        return this.asDNArray()
    else throw Exception("Cannot cast MultiArray to Ndarray of dimension n.")
}