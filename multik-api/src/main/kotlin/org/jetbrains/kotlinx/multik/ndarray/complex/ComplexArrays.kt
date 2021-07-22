package org.jetbrains.kotlinx.multik.ndarray.complex

public class ComplexFloatArray(public val size: Int = 0) {

    private val _size: Int = size * 2

    private val data: FloatArray = FloatArray(_size)

    public constructor(size: Int, init: (Int) -> ComplexFloat) : this(size) {
        for (i in 0 until size) {
            val (re, im) = init(i)
            val index = i * 2
            this.data[index] = re
            this.data[index + 1] = im
        }
    }

    public operator fun get(index: Int): ComplexFloat {
        checkElementIndex(index, size)
        return ComplexFloat(data[index], data[index + 1])
    }

    public operator fun set(index: Int, value: ComplexFloat): Unit {
        checkElementIndex(index, size)
        data[index] = value.re
        data[index + 1] = value.im
    }

    public fun getFlatArray(): FloatArray = data

    /** Creates an iterator over the elements of the array. */
    public operator fun iterator(): ComplexFloatIterator = iterator(this)

    override fun toString(): String {
        val sb = StringBuilder(2 + _size * 3)
        sb.append("[")
        var i = 0
        while (i < _size) {
            if (i > 0) sb.append(", ")
            sb.append("${data[i]} + ${data[i++]}i")
            i++
        }
        sb.append("]")
        return sb.toString()
    }
}

public class ComplexDoubleArray(public val size: Int = 0) {

    private val _size: Int = size * 2

    private val data: DoubleArray = DoubleArray(this._size)

    public constructor(size: Int, init: (Int) -> ComplexDouble) : this(size) {
        for (i in 0 until size) {
            val (re, im) = init(i)
            val index = i * 2
            this.data[index] = re
            this.data[index + 1] = im
        }
    }

    public operator fun get(index: Int): ComplexDouble {
        checkElementIndex(index, size)
        return ComplexDouble(data[index], data[index + 1])
    }

    public operator fun set(index: Int, value: ComplexDouble): Unit {
        checkElementIndex(index, size)
        data[index] = value.re
        data[index + 1] = value.im
    }

    public fun getFlatArray(): DoubleArray = data

    /** Creates an iterator over the elements of the array. */
    public operator fun iterator(): ComplexDoubleIterator = iterator(this)

    override fun toString(): String {
        floatArrayOf().isEmpty()
        val sb = StringBuilder(2 + _size * 3)
        sb.append("[")
        var i = 0
        while (i < _size) {
            if (i > 0) sb.append(", ")
            sb.append("${data[i]} + ${data[i++]}i")
            i++
        }
        sb.append("]")
        return sb.toString()
    }
}

private fun checkElementIndex(index: Int, size: Int) {
    if (index < 0 || index >= size) throw IndexOutOfBoundsException("index: $index, size: $size")
}