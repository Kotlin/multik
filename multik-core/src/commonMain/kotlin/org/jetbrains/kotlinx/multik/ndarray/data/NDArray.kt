package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.operations.concatenate

/** Convenience alias for a 1-dimensional [NDArray]. */
public typealias D1Array<T> = NDArray<T, D1>

/** Convenience alias for a 2-dimensional [NDArray]. */
public typealias D2Array<T> = NDArray<T, D2>

/** Convenience alias for a 3-dimensional [NDArray]. */
public typealias D3Array<T> = NDArray<T, D3>

/** Convenience alias for a 4-dimensional [NDArray]. */
public typealias D4Array<T> = NDArray<T, D4>

/**
 * Concrete multidimensional array backed by a flat [MemoryView] of primitive values.
 *
 * The array's multidimensional structure is defined by [offset], [shape], and [strides] over
 * a single contiguous buffer. Slicing, [transpose], and [reshape] return views that share the
 * underlying [data]; use [copy] or [deepCopy] to detach from the original buffer.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6]).reshape(2, 3)
 * // [[1, 2, 3],
 * //  [4, 5, 6]]
 * ```
 *
 * @param T the element type (e.g. [Int], [Double], [ComplexFloat]).
 * @param D the [Dimension] type ([D1], [D2], [D3], [D4], or [DN]).
 * @see [MultiArray]
 * @see [MutableMultiArray]
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
            return 0 until size
        }

    override val multiIndices: MultiIndexProgression get() = IntArray(dim.d)..IntArray(dim.d) { shape[it] - 1 }

    override fun isScalar(): Boolean = shape.isEmpty() || (shape.size == 1 && shape.first() == 1)

    public override fun isEmpty(): Boolean = size == 0

    public override fun isNotEmpty(): Boolean = !isEmpty()

    public override operator fun iterator(): Iterator<T> =
        if (consistent) this.data.iterator() else NDArrayIterator(data, offset, strides, shape)

    /**
     * Creates a new array with elements converted to the reified numeric type [E].
     *
     * ```
     * val intArray = mk.ndarray(mk[1, 2, 3])
     * val doubleArray = intArray.asType<Double>() // [1.0, 2.0, 3.0]
     * ```
     *
     * @param E the target numeric type.
     * @return a new [NDArray] with the converted elements.
     */
    public inline fun <reified E : Number> asType(): NDArray<E, D> {
        val dataType = DataType.ofKClass(E::class)
        return this.asType(dataType)
    }

    /**
     * Creates a new array with elements converted to the given [dataType].
     *
     * @param E the target numeric type.
     * @param dataType the target [DataType].
     * @return a new [NDArray] with the converted elements.
     */
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
        val newBase = if (consistent) this.base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return if (this.dim.d == 1 && this.shape.first() == dim1) {
            this as D1Array<T>
        } else {
            D1Array(newData, newOffset, intArrayOf(dim1), dim = D1, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int): D2Array<T> {
        val newShape = intArrayOf(dim1, dim2)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) this.base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return if (this.shape.contentEquals(newShape)) {
            this as D2Array<T>
        } else {
            D2Array(newData, newOffset, newShape, dim = D2, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int, dim3: Int): D3Array<T> {
        val newShape = intArrayOf(dim1, dim2, dim3)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 * dim3 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return if (this.shape.contentEquals(newShape)) {
            this as D3Array<T>
        } else {
            D3Array(newData, newOffset, newShape, dim = D3, base = newBase)
        }
    }

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T> {
        val newShape = intArrayOf(dim1, dim2, dim3, dim4)
        newShape.forEach { requirePositiveShape(it) }
        require(dim1 * dim2 * dim3 * dim4 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3, $dim4)" }

        // TODO(get rid of copying)
        val newData = if (consistent) this.data else this.deepCopy().data
        val newBase = if (consistent) this.base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return if (this.shape.contentEquals(newShape)) {
            this as D4Array<T>
        } else {
            D4Array(newData, newOffset, newShape, dim = D4, base = newBase)
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
        val newBase = if (consistent) this.base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return if (this.shape.contentEquals(newShape)) {
            this as NDArray<T, DN>
        } else {
            NDArray(newData, newOffset, newShape, dim = DN(newShape.size), base = newBase)
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
        val newBase = if (consistent) this.base ?: this else null
        val newOffset = if (consistent) this.offset else 0

        return NDArray(newData, newOffset, newShape.toIntArray(), dim = DN(newShape.size), base = newBase)
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

    /**
     * Casts this array to a [D1Array].
     *
     * @return this array typed as [D1Array].
     * @throws ClassCastException if this array is not 1-dimensional.
     */
    public fun asD1Array(): D1Array<T> {
        if (this.dim.d == 1) return this as D1Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 1.")
    }

    /**
     * Casts this array to a [D2Array].
     *
     * @return this array typed as [D2Array].
     * @throws ClassCastException if this array is not 2-dimensional.
     */
    public fun asD2Array(): D2Array<T> {
        if (this.dim.d == 2) return this as D2Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 2.")
    }

    /**
     * Casts this array to a [D3Array].
     *
     * @return this array typed as [D3Array].
     * @throws ClassCastException if this array is not 3-dimensional.
     */
    public fun asD3Array(): D3Array<T> {
        if (this.dim.d == 3) return this as D3Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 3.")
    }

    /**
     * Casts this array to a [D4Array].
     *
     * @return this array typed as [D4Array].
     * @throws ClassCastException if this array is not 4-dimensional.
     */
    public fun asD4Array(): D4Array<T> {
        if (this.dim.d == 4) return this as D4Array<T>
        else throw ClassCastException("Cannot cast NDArray of dimension ${this.dim.d} to NDArray of dimension 4.")
    }

    /**
     * Wraps this array with the [DN] dimension type.
     *
     * For arrays with dimension > 4, returns the array directly. For arrays with dimension 1–4,
     * creates a new view with [DN] as the dimension type.
     *
     * @return this array typed as `NDArray<T, DN>`.
     * @throws Exception if the array dimension is undefined (-1).
     */
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
        // TODO(workaround for value classes: ComplexFloat and ComplexDouble)
        // "https://youtrack.jetbrains.com/issue/KT-24874/Support-custom-equals-and-hashCode-for-value-classes"
        while (thIt.hasNext() && othIt.hasNext()) {
            val left = thIt.next()
            val right = othIt.next()
            when {
                left is ComplexFloat && right is ComplexFloat -> {
                    if (!(left.eq(right)))
                        return false
                }

                left is ComplexDouble && right is ComplexDouble -> {
                    if (!(left.eq(right)))
                        return false
                }

                else -> {
                    if (left != right)
                        return false
                }
            }
//            if (thIt.next() != othIt.next())
//                return false
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
