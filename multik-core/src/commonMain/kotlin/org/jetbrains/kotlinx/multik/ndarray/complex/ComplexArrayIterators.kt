package org.jetbrains.kotlinx.multik.ndarray.complex

/** Iterates over the elements of a [ComplexFloatArray] in index order. */
private class ArrayComplexFloatIterator(private val array: ComplexFloatArray) : ComplexFloatIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexFloat() = try { array[index++] } catch (e: IndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

/** Iterates over the elements of a [ComplexDoubleArray] in index order. */
private class ArrayComplexDoubleIterator(private val array: ComplexDoubleArray) : ComplexDoubleIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexDouble() = try { array[index++] } catch (e: IndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

/** Creates a [ComplexFloatIterator] over the given [array]. */
public fun iterator(array: ComplexFloatArray): ComplexFloatIterator = ArrayComplexFloatIterator(array)

/** Creates a [ComplexDoubleIterator] over the given [array]. */
public fun iterator(array: ComplexDoubleArray): ComplexDoubleIterator = ArrayComplexDoubleIterator(array)
