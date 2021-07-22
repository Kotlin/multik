package org.jetbrains.kotlinx.multik.ndarray.complex

private class ArrayComplexFloatIterator(private val array: ComplexFloatArray) : ComplexFloatIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexFloat() = try { array[index++] } catch (e: ArrayIndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

private class ArrayComplexDoubleIterator(private val array: ComplexDoubleArray) : ComplexDoubleIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexDouble() = try { array[index++] } catch (e: ArrayIndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

public fun iterator(array: ComplexFloatArray): ComplexFloatIterator = ArrayComplexFloatIterator(array)
public fun iterator(array: ComplexDoubleArray): ComplexDoubleIterator = ArrayComplexDoubleIterator(array)