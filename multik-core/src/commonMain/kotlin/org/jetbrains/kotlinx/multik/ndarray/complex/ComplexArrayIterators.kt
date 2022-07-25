/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

private class ArrayComplexFloatIterator(private val array: ComplexFloatArray) : ComplexFloatIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexFloat() = try { array[index++] } catch (e: IndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

private class ArrayComplexDoubleIterator(private val array: ComplexDoubleArray) : ComplexDoubleIterator() {
    private var index = 0
    override fun hasNext() = index < array.size
    override fun nextComplexDouble() = try { array[index++] } catch (e: IndexOutOfBoundsException) { index -= 1; throw NoSuchElementException(e.message) }
}

public fun iterator(array: ComplexFloatArray): ComplexFloatIterator = ArrayComplexFloatIterator(array)
public fun iterator(array: ComplexDoubleArray): ComplexDoubleIterator = ArrayComplexDoubleIterator(array)