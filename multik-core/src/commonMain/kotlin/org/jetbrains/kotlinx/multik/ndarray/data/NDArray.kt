/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.operations.concatenate

public typealias D1Array<T> = NDArray<T, D1>
public typealias D2Array<T> = NDArray<T, D2>
public typealias D3Array<T> = NDArray<T, D3>
public typealias D4Array<T> = NDArray<T, D4>

/**
 * A class that implements multidimensional arrays. This implementation is based on primitive arrays.
 * With the help of [offset], [shape], [strides] there is a multidimensionality representation
 * over a sequential homogeneous array.
 *
 * Native code uses `GetPrimitiveArrayCritical` for calculation.
 *
 * @param T type of stored values.
 * @param D dimension.
 */
public class NDArray<T, D : Dimension> constructor(
    data: ImmutableMemoryView<T>,
    public override val offset: Int = 0,
    public override val shape: IntArray,
    public override val strides: IntArray = computeStrides(shape),
    public override val dim: D,
    public override val base: MultiArray<T, out Dimension>? = null
) : MutableMultiArray<T, D> {

    init {
        check(shape.isNotEmpty()) { "Shape can't be empty." }
    }

    public override val data: MemoryView<T> = data as MemoryView<T>

    public override val size: Int get() = shape.fold(1, Int::times)

    public override val consistent: Boolean
        get() {
            return offset == 0 && size == data.size && strides.contentEquals(computeStrides(shape))
        }

    override val indices: IntRange
        get() {
            // todo?
//            if (dim.d != 1) throw IllegalStateException("NDArray of dimension ${dim.d}, use multiIndex.")
            return 0..size - 1
        }

    override val multiIndices: MultiIndexProgression get() = IntArray(dim.d)..IntArray(dim.d) { shape[it] - 1 }

    override fun isScalar(): Boolean = shape.isEmpty() || (shape.size == 1 && shape.first() == 1)

    public override fun isEmpty(): Boolean = size == 0

    public override fun isNotEmpty(): Boolean = !isEmpty()

    public override operator fun iterator(): Iterator<T> =
        if (consistent) this.data.iterator() else NDArrayIterator(data, offset, strides, shape)

    public inline fun <reified E : Number> asType(): NDArray<E, D> {
        val dataType = DataType.ofKClass(E::class)
        return this.asType(dataType)
    }

    //TODO ???
    public fun <E : Number> asType(dataType: DataType): NDArray<E, D> {
        val newData = initMemoryView(this.data.size, dataType) { this.data[it] as E }
        return NDArray(newData, this.offset, this.shape, this.strides, this.dim)
    }

    override fun copy(): NDArray<T, D> =
        NDArray(this.data.copyOf(), this.offset, this.shape.copyOf(), this.strides.copyOf(), this.dim)

    override fun deepCopy(): NDArray<T, D> {
        val data: MemoryView<T>

        if (consistent) {
            data = this.data.copyOf()
        } else {
            data = initMemoryView(this.size, this.dtype)
            var index = 0
            for (el in this)
                data[index++] = el
        }
        return NDArray(data, 0, this.shape.copyOf(), dim = this.dim)
    }

    override fun flatten(): MultiArray<T, D1> {
        val data = if (consistent) {
            data.copyOf()
        } else {
            val tmpData = initMemoryView<T>(size, dtype)
            var index = 0
            for (el in this) tmpData[index++] = el
            tmpData
        }
        return D1Array(data, 0, intArrayOf(size), dim = D1)
    }

    // TODO(strides? : view.reshape().reshape()?)
    override fun reshape(dim1: Int): D1Array<T> {
        // todo negative shape?
        requirePositiveShape(dim1)
        require(dim1 == size) { "Cannot reshape array of size $size into a new shape ($dim1)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return if (this.dim.d == 1 && this.shape.first() == dim1) {
            this as D1Array<T>
        } else {
            D1Array(newData, this.offset, intArrayOf(dim1), dim = D1, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int): D2Array<T> {
        val newShape = intArrayOf(dim1, dim2)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return if (this.shape.contentEquals(newShape)) {
            this as D2Array<T>
        } else {
            D2Array(newData, this.offset, newShape, dim = D2, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int, dim3: Int): D3Array<T> {
        val newShape = intArrayOf(dim1, dim2, dim3)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 * dim3 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return if (this.shape.contentEquals(newShape)) {
            this as D3Array<T>
        } else {
            D3Array(newData, this.offset, newShape, dim = D3, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T> {
        val newShape = intArrayOf(dim1, dim2, dim3, dim4)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 * dim3 * dim4 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3, $dim4)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return if (this.shape.contentEquals(newShape)) {
            this as D4Array<T>
        } else {
            D4Array(newData, this.offset, newShape, dim = D4, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): NDArray<T, DN> {
        val newShape = intArrayOf(dim1, dim2, dim3, dim4) + dims
        newShape.forEach { requirePositiveShape(it) }
        require(newShape.fold(1, Int::times) == size) {
            "Cannot reshape array of size $size into a new shape ${newShape.joinToString(prefix = "(", postfix = ")")}"
        }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return if (this.shape.contentEquals(newShape)) {
            this as NDArray<T, DN>
        } else {
            NDArray(newData, this.offset, newShape, dim = DN(newShape.size), base = newBase)
        }
    }

    override fun transpose(vararg axes: Int): NDArray<T, D> {
        require(axes.isEmpty() || axes.size == dim.d) { "All dimensions must be indicated." }
        for (axis in axes) require(axis in 0 until dim.d) { "Dimension must be from 0 to ${dim.d}." }
        require(axes.toSet().size == axes.size) { "The specified dimensions must be unique." }
        if (dim.d == 1) return NDArray(this.data, this.offset, this.shape, this.strides, this.dim)
        val newShape: IntArray
        val newStrides: IntArray
        if (axes.isEmpty()) {
            newShape = this.shape.reversedArray()
            newStrides = this.strides.reversedArray()
        } else {
            newShape = IntArray(this.shape.size)
            newStrides = IntArray(this.strides.size)
            for ((i, axis) in axes.withIndex()) {
                newShape[i] = this.shape[axis]
                newStrides[i] = this.strides[axis]
            }
        }
        return NDArray(this.data, this.offset, newShape, newStrides, this.dim, base = base ?: this)
    }

    override fun squeeze(vararg axes: Int): NDArray<T, DN> {
        val cutAxes = if (axes.isEmpty()) {
            shape.withIndex().filter { it.value == 1 }.map { it.index }
        } else {
            require(axes.all { shape[it] == 1 }) { "Cannot select an axis to squeeze out which has size not equal to one." }
            axes.toList()
        }
        val newShape = this.shape.sliceArray(this.shape.indices - cutAxes)
        return NDArray(this.data, this.offset, newShape, dim = DN(newShape.size), base = base ?: this)
    }

    override fun unsqueeze(vararg axes: Int): NDArray<T, DN> {
        val newShape = shape.toMutableList()
        for (axis in axes.sorted()) {
            newShape.add(axis, 1)
        }
        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null

        return NDArray(newData, this.offset, newShape.toIntArray(), dim = DN(newShape.size), base = newBase)
    }

    override infix fun cat(other: MultiArray<T, D>): NDArray<T, D> =
        cat(listOf(other), 0)

    override fun cat(other: MultiArray<T, D>, axis: Int): NDArray<T, D> =
        cat(listOf(other), axis)

    override fun cat(other: List<MultiArray<T, D>>, axis: Int): NDArray<T, D> {
        val actualAxis = actualAxis(axis)
        require(actualAxis in 0 until dim.d) { "Axis $axis is out of bounds for array of dimension $dim" }
        val arr = other.first()
        require(
            this.shape.withIndex()
                .all { it.index == axis || it.value == arr.shape[it.index] }) { "All dimensions of input arrays for the concatenation axis must match exactly." }
        val newShape = this.shape.copyOf()
        newShape[actualAxis] = this.shape[actualAxis] + other.sumOf { it.shape[actualAxis] }
        val newSize = this.size + other.sumOf { it.size }
        val arrays = other.toMutableList().also { it.add(0, this) }
        val result = NDArray<T, D>(initMemoryView(newSize, dtype), 0, newShape, dim = dim)
        concatenate(arrays, result, actualAxis)
        return result
    }

    //todo extensions
    public fun asD1Array(): D1Array<T> {
        if (this.dim.d == 1) return this as D1Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 1.")
    }

    //todo
    public fun asD2Array(): D2Array<T> {
        if (this.dim.d == 2) return this as D2Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 2.")
    }

    public fun asD3Array(): D3Array<T> {
        if (this.dim.d == 3) return this as D3Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 3.")
    }

    public fun asD4Array(): D4Array<T> {
        if (this.dim.d == 4) return this as D4Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 4.")
    }

    public fun asDNArray(): NDArray<T, DN> {
        if (this.dim.d == -1) throw Exception("Array dimension is undefined")
        if (this.dim.d > 4) return this as NDArray<T, DN>

        return NDArray(this.data, this.offset, this.shape, this.strides, DN(this.dim.d), base = base ?: this)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as NDArray<*, *>

        if (size != other.size) return false
        if (!shape.contentEquals(other.shape)) return false
        if (dtype != other.dtype) return false
        if (dim != other.dim) return false

        val thIt = this.iterator()
        val othIt = other.iterator()
        while (thIt.hasNext() && othIt.hasNext()) {
            if (thIt.next() != othIt.next())
                return false
        }

        return true
    }

    override fun hashCode(): Int {
        var result = 1
        for (el in this) {
            result = 31 * result + el.hashCode()
        }
        return result
    }

    override fun toString(): String {
        return when (dim.d) {
            1 -> buildString {
                this@NDArray as NDArray<T, D1>
                append('[')
                for (i in 0 until shape.first()) {
                    append(this@NDArray[i])
                    if (i < shape.first() - 1)
                        append(", ")
                }
                append(']')
            }

            2 -> buildString {
                this@NDArray as NDArray<T, D2>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append(this@NDArray[ax0, ax1])
                        if (ax1 < shape[1] - 1)
                            append(", ")
                    }
                    append(']')
                    if (ax0 < shape[0] - 1)
                        append(",\n")
                }
                append(']')
            }

            3 -> buildString {
                this@NDArray as NDArray<T, D3>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append('[')
                        for (ax2 in 0 until shape[2]) {
                            append(this@NDArray[ax0, ax1, ax2])
                            if (ax2 < shape[2] - 1)
                                append(", ")
                        }
                        append(']')
                        if (ax1 < shape[1] - 1)
                            append(",\n")
                    }
                    append(']')
                    if (ax0 < shape[0] - 1)
                        append(",\n\n")
                }
                append(']')
            }

            4 -> buildString {
                this@NDArray as NDArray<T, D4>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append('[')
                        for (ax2 in 0 until shape[2]) {
                            append('[')
                            for (ax3 in 0 until shape[3]) {
                                append(this@NDArray[ax0, ax1, ax2, ax3])
                                if (ax3 < shape[3] - 1)
                                    append(", ")
                            }
                            append(']')
                            if (ax2 < shape[2] - 1)
                                append(",\n")
                        }
                        append(']')
                        if (ax1 < shape[1] - 1)
                            append(",\n\n")
                    }
                    append(']')
                    if (ax0 < shape[0] - 1)
                        append(",\n\n\n")
                }
                append(']')
            }

            else -> buildString {
                this@NDArray as NDArray<*, DN>
                append('[')
                for (ind in 0 until shape.first()) {
                    append(this@NDArray.V[ind].toString())
                    if (ind < shape.first() - 1) {
                        val newLine = "\n".repeat(dim.d - 1)
                        append(",$newLine")
                    }
                }
                append(']')
            }
        }
    }
}
