package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * An iterator over a sequence of [ComplexFloat] values.
 *
 * Subclasses must implement [nextComplexFloat] to return the next element without boxing.
 */
public abstract class ComplexFloatIterator : Iterator<ComplexFloat> {
    final override fun next(): ComplexFloat = nextComplexFloat()

    /** Returns the next [ComplexFloat] element in the iteration. */
    public abstract fun nextComplexFloat(): ComplexFloat
}

/**
 * An iterator over a sequence of [ComplexDouble] values.
 *
 * Subclasses must implement [nextComplexDouble] to return the next element without boxing.
 */
public abstract class ComplexDoubleIterator : Iterator<ComplexDouble> {
    final override fun next(): ComplexDouble = nextComplexDouble()

    /** Returns the next [ComplexDouble] element in the iteration. */
    public abstract fun nextComplexDouble(): ComplexDouble
}
