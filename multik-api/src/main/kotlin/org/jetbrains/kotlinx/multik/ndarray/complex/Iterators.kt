package org.jetbrains.kotlinx.multik.ndarray.complex

public abstract class ComplexFloatIterator : Iterator<ComplexFloat> {
    final override fun next(): ComplexFloat = nextComplexFloat()

    public abstract fun nextComplexFloat(): ComplexFloat
}

public abstract class ComplexDoubleIterator : Iterator<ComplexDouble> {
    final override fun next(): ComplexDouble = nextComplexDouble()

    public abstract fun nextComplexDouble(): ComplexDouble
}