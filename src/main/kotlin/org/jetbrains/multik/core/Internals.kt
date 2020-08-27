package org.jetbrains.multik.core

@PublishedApi
@Suppress("NOTHING_TO_INLINE")
internal inline fun requireDimension(dim: Dimension, shapeSize: Int) {
    require(dim.d == shapeSize) { "Dimension doesn't match the size of the shape: dimension (${dim.d}) != $shapeSize shape size." }
}

@PublishedApi
@Suppress("NOTHING_TO_INLINE")
internal inline fun requireShapeEmpty(shape: IntArray) {
    require(shape.isNotEmpty()) { "Shape cannot be empty." }
}

@Suppress("NOTHING_TO_INLINE")
internal inline fun requireElementsWithShape(elementSize: Int, shapeSize: Int) {
    require(elementSize == shapeSize) { "The number of elements doen't match the shape: $elementSize!=$shapeSize" }
}

@Suppress("NOTHING_TO_INLINE")
internal inline fun requireArraySizes(rightSize: Int, otherSize: Int) {
    require(rightSize == otherSize) { "Array sizes don't match: (right operand size) ${rightSize}!= ${otherSize} (left operand size)" }
}

@Suppress("NOTHING_TO_INLINE")
internal inline fun requirePositiveShape(dim: Int) {
    require(dim > 0) { "Shape must be positive but was $dim." }
}

internal fun computeStrides(shape: IntArray): IntArray = shape.clone().apply {
    this[this.lastIndex] = 1
    for (i in this.lastIndex - 1 downTo 0) {
        this[i] = this[i + 1] * shape[i + 1]
    }
}

@Suppress("NOTHING_TO_INLINE")
internal inline fun zeroNumber(dtype: DataType): Number = when (dtype.nativeCode) {
    1 -> 0.toByte()
    2 -> 0.toShort()
    3 -> 0
    4 -> 0L
    5 -> 0f
    6 -> 0.0
    else -> throw Exception("Type not defined.")
}

@PublishedApi
@Suppress("IMPLICIT_CAST_TO_ANY")
internal inline fun <reified T : Number> Number.toPrimitiveType(): T = when (T::class) {
    Byte::class -> this.toByte()
    Short::class -> this.toShort()
    Int::class -> this.toInt()
    Long::class -> this.toLong()
    Float::class -> this.toFloat()
    Double::class -> this.toDouble()
    else -> throw Exception("Type not defined.")
} as T

@Suppress("IMPLICIT_CAST_TO_ANY", "NOTHING_TO_INLINE", "UNCHECKED_CAST")
internal inline fun <T : Number> Number.toPrimitiveType(dtype: DataType): T = when (dtype.nativeCode) {
    1 -> this.toByte()
    2 -> this.toShort()
    3 -> this.toInt()
    4 -> this.toLong()
    5 -> this.toFloat()
    6 -> this.toDouble()
    else -> throw Exception("Type not defined.")
} as T

internal operator fun <T : Number> Number.compareTo(other: T): Int = (this - other).toInt()

internal fun IntArray.remove(pos: Int): IntArray = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
}

internal fun IntArray.removeAll(indices: List<Int>): IntArray = when {
    indices.isEmpty() -> this
    indices.size == 1 -> remove(indices.first())
    else -> this.filterIndexed { index, _ -> index !in indices }.toIntArray()
}
