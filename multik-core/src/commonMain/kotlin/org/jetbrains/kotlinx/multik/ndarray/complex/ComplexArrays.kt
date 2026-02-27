package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * An array of [ComplexFloat] values stored as interleaved real/imaginary pairs in a flat [FloatArray].
 *
 * Each complex element occupies two consecutive floats: `[re₀, im₀, re₁, im₁, …]`.
 * The [size] property returns the number of complex elements (half the length of the backing array).
 *
 * ```
 * val arr = ComplexFloatArray(3) { ComplexFloat(it.toFloat(), 0f) }
 * arr[1] // 1.0+(0.0)i
 * ```
 *
 * @property size the number of complex elements in this array.
 * @see ComplexDoubleArray
 */
public class ComplexFloatArray(public val size: Int = 0) {

    private val _size: Int = size * 2

    private val data: FloatArray = FloatArray(_size)

    /**
     * Creates a [ComplexFloatArray] of the given [size] where each element is produced by the [init] function.
     *
     * @param size the number of complex elements.
     * @param init a function that returns the [ComplexFloat] value for each index.
     */
    public constructor(size: Int, init: (Int) -> ComplexFloat) : this(size) {
        for (i in 0 until size) {
            val (re, im) = init(i)
            val index = i * 2
            this.data[index] = re
            this.data[index + 1] = im
        }
    }

    /**
     * Returns the complex element at the given [index].
     *
     * @throws IndexOutOfBoundsException if [index] is negative or >= [size].
     */
    public operator fun get(index: Int): ComplexFloat {
        checkElementIndex(index, size)
        val i = index shl 1
        return ComplexFloat(data[i], data[i + 1])
    }

    /**
     * Sets the complex element at the given [index] to [value].
     *
     * @throws IndexOutOfBoundsException if [index] is negative or >= [size].
     */
    public operator fun set(index: Int, value: ComplexFloat): Unit {
        checkElementIndex(index, size)
        val i = index shl 1
        data[i] = value.re
        data[i + 1] = value.im
    }

    /**
     * Returns the backing interleaved [FloatArray] of length `size * 2`.
     *
     * Modifications to the returned array are reflected in this [ComplexFloatArray] and vice versa.
     */
    public fun getFlatArray(): FloatArray = data

    /** Creates an iterator over the elements of the array. */
    public operator fun iterator(): ComplexFloatIterator = iterator(this)

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexFloatArray -> this.contentEquals(other)
        else -> false
    }

    override fun hashCode(): Int = this.contentHashCode()

    @Suppress("DuplicatedCode")
    override fun toString(): String {
        val sb = StringBuilder(2 + _size * 3)
        sb.append("[")
        var i = 0
        while (i < _size) {
            if (i > 0) sb.append(", ")
            sb.append("${data[i]} + ${data[++i]}i")
            i++
        }
        sb.append("]")
        return sb.toString()
    }
}

/**
 * An array of [ComplexDouble] values stored as interleaved real/imaginary pairs in a flat [DoubleArray].
 *
 * Each complex element occupies two consecutive doubles: `[re₀, im₀, re₁, im₁, …]`.
 * The [size] property returns the number of complex elements (half the length of the backing array).
 *
 * ```
 * val arr = ComplexDoubleArray(3) { ComplexDouble(it.toDouble(), 0.0) }
 * arr[1] // 1.0+(0.0)i
 * ```
 *
 * @property size the number of complex elements in this array.
 * @see ComplexFloatArray
 */
public class ComplexDoubleArray(public val size: Int = 0) {

    private val _size: Int = size * 2

    private val data: DoubleArray = DoubleArray(this._size)

    /**
     * Creates a [ComplexDoubleArray] of the given [size] where each element is produced by the [init] function.
     *
     * @param size the number of complex elements.
     * @param init a function that returns the [ComplexDouble] value for each index.
     */
    public constructor(size: Int, init: (Int) -> ComplexDouble) : this(size) {
        for (i in 0 until size) {
            val (re, im) = init(i)
            val index = i * 2
            this.data[index] = re
            this.data[index + 1] = im
        }
    }

    /**
     * Returns the complex element at the given [index].
     *
     * @throws IndexOutOfBoundsException if [index] is negative or >= [size].
     */
    public operator fun get(index: Int): ComplexDouble {
        checkElementIndex(index, size)
        val i = index shl 1
        return ComplexDouble(data[i], data[i + 1])
    }

    /**
     * Sets the complex element at the given [index] to [value].
     *
     * @throws IndexOutOfBoundsException if [index] is negative or >= [size].
     */
    public operator fun set(index: Int, value: ComplexDouble): Unit {
        checkElementIndex(index, size)
        val i = index shl 1
        data[i] = value.re
        data[i + 1] = value.im
    }

    /**
     * Returns the backing interleaved [DoubleArray] of length `size * 2`.
     *
     * Modifications to the returned array are reflected in this [ComplexDoubleArray] and vice versa.
     */
    public fun getFlatArray(): DoubleArray = data

    /** Creates an iterator over the elements of the array. */
    public operator fun iterator(): ComplexDoubleIterator = iterator(this)

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexDoubleArray -> this.contentEquals(other)
        else -> false
    }

    override fun hashCode(): Int = this.contentHashCode()

    @Suppress("DuplicatedCode")
    override fun toString(): String {
        val sb = StringBuilder(2 + _size * 3)
        sb.append("[")
        var i = 0
        while (i < _size) {
            if (i > 0) sb.append(", ")
            sb.append("${data[i]} + ${data[++i]}i")
            i++
        }
        sb.append("]")
        return sb.toString()
    }
}

private fun checkElementIndex(index: Int, size: Int) {
    if (index < 0 || index >= size) throw IndexOutOfBoundsException("index: $index, size: $size")
}
