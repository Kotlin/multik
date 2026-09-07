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
@Suppress( "nothing_to_inline")
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
@Suppress( "nothing_to_inline")
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
@Suppress( "nothing_to_inline")
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
@Suppress( "nothing_to_inline")
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
@Suppress( "nothing_to_inline")
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
@Suppress( "nothing_to_inline")
internal inline fun requireEqualShape(left: IntArray, right: IntArray) {
    require(left.contentEquals(right)) { "Array shapes don't match: ${left.contentToString()} != ${right.contentToString()}" }
}

/**
 * Checks if the given dimension is positive or not. Throws an IllegalArgumentException if the shape is not positive.
 *
 * @param dim an integer representing the dimension of the shape.
 * @throws IllegalArgumentException if the shape dimension is not positive.
 */
@Suppress( "nothing_to_inline")
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

/**
 * Computes strides addressing [newShape] over the memory described by [shape] and [strides],
 * without moving any element.
 *
 * A shape change can be expressed as a view whenever every group of old axes that the new shape
 * merges is contiguous with respect to itself. Axes of size one are ignored, because a single
 * element is reachable regardless of the stride assigned to its axis. This is the rule NumPy uses,
 * so a copy is only needed when the requested layout genuinely cannot be strided over the existing
 * buffer — for example when reshaping a transposed array to a shape that mixes its axes.
 *
 * @param shape the current shape.
 * @param strides the current strides, in elements.
 * @param newShape the requested shape; must describe the same number of elements as [shape].
 * @return strides for [newShape] over the same memory, or `null` when the elements must be copied.
 */
internal fun reshapeStrides(shape: IntArray, strides: IntArray, newShape: IntArray): IntArray? {
    if (newShape.isEmpty()) return IntArray(0)
    // An empty array addresses no elements, so the packed strides of the new shape always fit.
    if (shape.any { it == 0 }) return computeStrides(newShape)

    // Drop axes of size one: their stride is never used and would otherwise break up
    // groups of axes that are in fact contiguous.
    var oldNd = 0
    val oldShape = IntArray(shape.size)
    val oldStrides = IntArray(shape.size)
    for (axis in shape.indices) {
        if (shape[axis] != 1) {
            oldShape[oldNd] = shape[axis]
            oldStrides[oldNd] = strides[axis]
            oldNd++
        }
    }

    val newStrides = IntArray(newShape.size)
    // [oi, oj) and [ni, nj) delimit the groups of old and new axes holding the same elements.
    var oi = 0
    var oj = 1
    var ni = 0
    var nj = 1
    while (ni < newShape.size && oi < oldNd) {
        var newSize = newShape[ni]
        var oldSize = oldShape[oi]
        while (newSize != oldSize) {
            if (newSize < oldSize) newSize *= newShape[nj++] else oldSize *= oldShape[oj++]
        }

        // Merging old axes is only possible when each one packs the next one exactly.
        for (axis in oi until oj - 1) {
            if (oldStrides[axis] != oldShape[axis + 1] * oldStrides[axis + 1]) return null
        }

        newStrides[nj - 1] = oldStrides[oj - 1]
        for (axis in nj - 1 downTo ni + 1) {
            newStrides[axis - 1] = newStrides[axis] * newShape[axis]
        }

        ni = nj++
        oi = oj++
    }

    // Trailing axes of size one in the new shape; their stride is never used.
    val lastStride = if (ni > 0) newStrides[ni - 1] else 1
    for (axis in ni until newShape.size) newStrides[axis] = lastStride

    return newStrides
}

/**
 * Returns an array of [newShape] over this array's elements, sharing memory whenever possible.
 *
 * The result is a view whose [base][MultiArray.base] is the owner of this array's buffer when
 * [reshapeStrides] can express [newShape] over the current layout, and a freshly packed copy
 * otherwise.
 *
 * @param newShape the requested shape; callers must have checked that it holds [MultiArray.size] elements.
 * @param newDim the [Dimension] matching [newShape].
 */
internal fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.reshapeTo(
    newShape: IntArray, newDim: O
): NDArray<T, O> {
    val newStrides = reshapeStrides(shape, strides, newShape)
    return if (newStrides != null) {
        NDArray(data, offset, newShape, newStrides, newDim, base ?: this)
    } else {
        NDArray(deepCopy().data, 0, newShape, computeStrides(newShape), newDim)
    }
}

/** Converts this [Number] to the reified primitive type [T]. */
@PublishedApi
internal inline fun <reified T : Number> Number.toPrimitiveType(): T = when (T::class) {
    Byte::class -> this.toByte()
    Short::class -> this.toShort()
    Int::class -> this.toInt()
    Long::class -> this.toLong()
    Float::class -> this.toFloat()
    Double::class -> this.toDouble()
    else -> throw IllegalArgumentException(
        "Cannot convert $this to ${T::class.simpleName}: expected Byte, Short, Int, Long, Float or Double."
    )
} as T

/**
 * Converts this [Number] to the numeric type described by [dtype].
 *
 * @param dtype the target [DataType] (must be a real numeric type, not complex).
 * @return this value converted to the target type.
 * @throws IllegalArgumentException if [dtype] is not a real numeric type.
 */
@Suppress("UNCHECKED_CAST")
public fun <T : Number> Number.toPrimitiveType(dtype: DataType): T = when (dtype.nativeCode) {
    1 -> this.toByte()
    2 -> this.toShort()
    3 -> this.toInt()
    4 -> this.toLong()
    5 -> this.toFloat()
    6 -> this.toDouble()
    else -> throw IllegalArgumentException(
        "Cannot convert $this to ${dtype.name}: expected a real numeric type, not a complex one."
    )
} as T

/**
 * Compares this [Number] to [other] using type-specific comparison when both have the same runtime type,
 * falling back to [Double] comparison otherwise.
 */
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
