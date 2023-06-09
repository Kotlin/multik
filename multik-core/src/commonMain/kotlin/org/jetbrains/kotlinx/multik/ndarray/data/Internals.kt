/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data


/**
 * Checks if the given index is within the bounds of the given axis and the size of the shape.
 *
 * @param value the boolean value representing whether the index is within bounds
 * @param index the integer value representing the index to check
 * @param axis the integer value representing the axis dimension to check against
 * @param size the integer value representing the size of the shape on the given axis dimension
 *
 * @throws IndexOutOfBoundsException when the index is out of bounds for the given axis and size
 */
@PublishedApi
internal inline fun checkBounds(value: Boolean, index: Int, axis: Int, size: Int) {
    if (!value) {
        throw IndexOutOfBoundsException("Index $index is out of bounds shape dimension $axis with size $size")
    }
}

/**
 * Checks if the given dimension matches the provided shape size, or if the dimension is greater than 4
 * and shape size is greater than 4.
 *
 * @param dim the input dimension object to check.
 * @param shapeSize the size of the shape to compare with.
 * @throws IllegalArgumentException if the dimension doesn't match the size of the shape.
 */
@PublishedApi
internal inline fun requireDimension(dim: Dimension, shapeSize: Int) {
    require(dim.d == shapeSize || (dim.d > 4 && shapeSize > 4))
    { "Dimension doesn't match the size of the shape: dimension (${dim.d}) != $shapeSize shape size." }
}

/**
 * Check if the given shape is empty.
 *
 * @param shape An array of integers representing the shape to be checked.
 * @throws IllegalArgumentException if the given shape is empty.
 */
@PublishedApi
internal inline fun requireShapeEmpty(shape: IntArray) {
    require(shape.isNotEmpty()) { "Shape cannot be empty." }
}

/**
 * Checks if the number of elements matches the specified shape.
 *
 * @param elementSize the number of elements in the element list
 * @param shapeSize the size of the given shape
 * @throws IllegalArgumentException if the number of elements doesn't match the shape
 */
internal inline fun requireElementsWithShape(elementSize: Int, shapeSize: Int) {
    require(elementSize == shapeSize) { "The number of elements doesn't match the shape: $elementSize!=$shapeSize" }
}

/**
 * Asserts that two array sizes are equal.
 *
 * @param rightSize the size of the right operand array
 * @param otherSize the size of the left operand array
 *
 * @throws IllegalArgumentException if the two sizes don't match
 */
internal inline fun requireArraySizes(rightSize: Int, otherSize: Int) {
    require(rightSize == otherSize) { "Array sizes don't match: (right operand size) $rightSize != $otherSize (left operand size)" }
}

/**
 * Checks if two given integer arrays have equal shape.
 *
 * @param left the first integer array to compare
 * @param right the second integer array to compare
 * @throws IllegalArgumentException if the shapes of the arrays do not match
 */
internal inline fun requireEqualShape(left: IntArray, right: IntArray) {
    require(left.contentEquals(right)) { "Array shapes don't match: ${left.contentToString()} != ${right.contentToString()}" }
}

/**
 * Checks if the given dimension is positive or not. Throws an IllegalArgumentException if the shape is not positive.
 *
 * @param dim an integer representing the dimension of the shape.
 * @throws IllegalArgumentException if the shape dimension is not positive.
 */
internal inline fun requirePositiveShape(dim: Int) {
    require(dim > 0) { "Shape must be positive but was $dim." }
}

/**
 * Computes the strides for a multidimensional array given the shape.
 *
 * @param shape an array representing the shape of the multidimensional array
 * @return an integer array containing the strides of the multidimensional array
 */
internal fun computeStrides(shape: IntArray): IntArray = shape.copyOf().apply {
    this[this.lastIndex] = 1
    for (i in this.lastIndex - 1 downTo 0) {
        this[i] = this[i + 1] * shape[i + 1]
    }
}

@PublishedApi
internal inline fun <reified T : Number> Number.toPrimitiveType(): T = when (T::class) {
    Byte::class -> this.toByte()
    Short::class -> this.toShort()
    Int::class -> this.toInt()
    Long::class -> this.toLong()
    Float::class -> this.toFloat()
    Double::class -> this.toDouble()
    else -> throw Exception("Type not defined.")
} as T

@Suppress("UNCHECKED_CAST")
public fun <T : Number> Number.toPrimitiveType(dtype: DataType): T = when (dtype.nativeCode) {
    1 -> this.toByte()
    2 -> this.toShort()
    3 -> this.toInt()
    4 -> this.toLong()
    5 -> this.toFloat()
    6 -> this.toDouble()
    else -> throw Exception("Type not defined.")
} as T

public operator fun <T : Number> Number.compareTo(other: T): Int {
    return when {
        this is Float && other is Float -> this.compareTo(other)
        this is Double && other is Double -> this.compareTo(other)
        this is Int && other is Int -> this.compareTo(other)
        this is Long && other is Long -> this.compareTo(other)
        this is Short && other is Short -> this.compareTo(other)
        this is Byte && other is Byte -> this.compareTo(other)
        else -> this.toDouble().compareTo(other.toDouble())
    }
}

/**
 * Returns the actual axis index by converting a negative index to positive index relative to the array dimensions
 *
 * @param axis the index of the axis to retrieve
 * @return the actual axis index
 */
internal fun MultiArray<*, *>.actualAxis(axis: Int): Int {
    return if (axis < 0) dim.d + axis else axis
}

/**
 * Removes the element at the specified position in this IntArray.
 *
 * @param pos the position of the element to be removed
 * @return the new IntArray with the element removed
 */
@PublishedApi
internal fun IntArray.remove(pos: Int): IntArray = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
}

/**
 * Removes elements from the array with indices specified in the given list.
 *
 * @param indices the list of element indices to be removed from the array.
 * @return the new array with requested elements removed, or the original array if the list is empty.
 */
internal fun IntArray.removeAll(indices: List<Int>): IntArray = when {
    indices.isEmpty() -> this
    indices.size == 1 -> remove(indices.first())
    else -> this.filterIndexed { index, _ -> index !in indices }.toIntArray()
}
