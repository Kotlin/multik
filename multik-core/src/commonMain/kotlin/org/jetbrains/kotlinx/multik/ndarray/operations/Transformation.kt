package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.D4
import org.jetbrains.kotlinx.multik.ndarray.data.D4Array
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryView
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.actualAxis
import org.jetbrains.kotlinx.multik.ndarray.data.dimensionOf
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.initMemoryView
import kotlin.jvm.JvmName

/**
 * Appends the given [value] elements to this array, returning a new flattened 1D array.
 *
 * The source array is flattened before appending. The original is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * a.append(4, 5) // [1, 2, 3, 4, 5]
 * ```
 *
 * @param value elements to append.
 * @return a new [D1Array] with size `this.size + value.size`.
 * @see [cat] for concatenation along a specific axis.
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(vararg value: T): D1Array<T> {
    val newSize = this.size + value.size
    val data = this.copyFromTwoArrays(value.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 * Appends all elements of [arr] to this array, returning a new flattened 1D array.
 *
 * Both arrays are flattened before concatenation. The originals are not modified.
 *
 * @param arr the array whose elements are appended.
 * @return a new [D1Array] with size `this.size + arr.size`.
 * @see [cat] for concatenation along a specific axis.
 */
public infix fun <T, D : Dimension, ID : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, ID>): D1Array<T> {
    val newSize = this.size + arr.size
    val data = this.copyFromTwoArrays(arr.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 * Concatenates [arr] to this array along the given [axis], returning a new array.
 *
 * Both arrays must have the same shape except along [axis]. Delegates to [cat].
 *
 * @param arr the array to concatenate.
 * @param axis the axis along which to join.
 * @return a new [NDArray] with the two arrays joined along [axis].
 * @see [cat]
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, D>, axis: Int): NDArray<T, D> =
    this.cat(arr, axis)

/**
 * Stacks 1D arrays along a new [axis], returning a 2D array.
 *
 * All input arrays must have the same shape. The result has one more dimension
 * than the inputs, with the new dimension at position [axis].
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * val b = mk.ndarray(mk[4, 5, 6])
 * mk.stack(a, b) // [[1, 2, 3], [4, 5, 6]]  shape: (2, 3)
 * ```
 *
 * @param arr the arrays to stack (must all have the same shape).
 * @param axis position of the new axis in the result (default 0).
 * @return a new [NDArray] with rank = input rank + 1.
 * @throws IllegalArgumentException if fewer than 2 arrays are provided, shapes differ, or [axis] is out of bounds.
 * @see [cat] for joining along an existing axis.
 */
@JvmName("stackD1")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D1>, axis: Int = 0): NDArray<T, D2> =
    stack(arr.toList(), axis)

/** Stacks a list of 1D arrays along a new [axis], returning a 2D array. */
@JvmName("stackD1")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D1>>, axis: Int = 0): NDArray<T, D2> =
    stackArrays(arrays, axis)

/** Stacks 2D arrays along a new [axis], returning a 3D array. */
@JvmName("stackD2")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D2>, axis: Int = 0): NDArray<T, D3> =
    stack(arr.toList(), axis)

/** Stacks a list of 2D arrays along a new [axis], returning a 3D array. */
@JvmName("stackD2")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D2>>, axis: Int = 0): NDArray<T, D3> =
    stackArrays(arrays, axis)

/** Stacks 3D arrays along a new [axis], returning a 4D array. */
@JvmName("stackD3")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D3>, axis: Int = 0): NDArray<T, D4> =
    stack(arr.toList(), axis)

/** Stacks a list of 3D arrays along a new [axis], returning a 4D array. */
@JvmName("stackD3")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D3>>, axis: Int = 0): NDArray<T, D4> =
    stackArrays(arrays, axis)

/** Stacks 4D arrays along a new [axis], returning an N-dimensional array. */
@JvmName("stackD4")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D4>, axis: Int = 0): NDArray<T, DN> =
    stack(arr.toList(), axis)

/** Stacks a list of 4D arrays along a new [axis], returning an N-dimensional array. */
@JvmName("stackD4")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D4>>, axis: Int = 0): NDArray<T, DN> =
    stackArrays(arrays, axis)

/** Implements stack by inserting a new axis at [axis] and copying elements from [arrays] into the result. */
private fun <T, ID : Dimension, OD : Dimension> stackArrays(
    arrays: List<MultiArray<T, ID>>,
    axis: Int = 0
): NDArray<T, OD> {
    require(arrays.isNotEmpty() && arrays.size > 1) { "Arrays list is empty or contains one element." }
    val firstArray = arrays.first()
    val actualAxis = firstArray.actualAxis(axis)
    val dim = dimensionOf<OD>(firstArray.dim.d + 1)
    require(axis in 0..firstArray.dim.d) { "Axis $axis is out of bounds for array of dimension $dim" }
    val arrShape = firstArray.shape
    require(arrays.all { it.shape.contentEquals(arrShape) }) { "Arrays must be of the same shape." }

    val shape = arrShape.toMutableList().apply { add(actualAxis, arrays.size) }.toIntArray()
    val size = shape.fold(1, Int::times)
    val result = NDArray(initMemoryView<T>(size, arrays.first().dtype), shape = shape, dim = dim)
    concatenate(arrays, result, axis = axis)
    return result
}

/**
 * Repeats this array's elements [n] times, returning a flat 1D array.
 *
 * The original array is flattened, then the flat content is tiled [n] times.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * a.repeat(2) // [1, 2, 3, 1, 2, 3]
 * ```
 *
 * @param n the number of repetitions (must be >= 1).
 * @return a new [D1Array] with size `this.size * n`.
 * @throws IllegalArgumentException if [n] < 1.
 */
public fun <T, D : Dimension> MultiArray<T, D>.repeat(n: Int): D1Array<T> {
    require(n >= 1) { "The number of repetitions must be more than one." }
    val data = initMemoryView<T>(size * n, dtype)
    if (consistent) {
        this.data.copyInto(data)
    } else {
        var index = 0
        for (el in this)
            data[index++] = el
    }

    for (i in size until (size * n) step size) {
        val startIndex = i - size
        val endIndex = i
        val startIndexComplex = startIndex * 2
        val endIndexComplex = startIndexComplex + (size * 2)

        when (this.dtype) {
            DataType.ComplexFloatDataType ->
                data.getFloatArray().copyInto(data.getFloatArray(), i * 2, startIndexComplex, endIndexComplex)

            DataType.ComplexDoubleDataType ->
                data.getDoubleArray().copyInto(data.getDoubleArray(), i * 2, startIndexComplex, endIndexComplex)

            else -> data.copyInto(data, i, i - size, endIndex)
        }
    }
    return D1Array(data, shape = intArrayOf(size * n), dim = D1)
}


/** Copies this array's elements followed by elements from [iter] into a new [MemoryView] of the given [size]. */
internal fun <T, D : Dimension> MultiArray<T, D>.copyFromTwoArrays(iter: Iterator<T>, size: Int): MemoryView<T> {
    val data = initMemoryView<T>(size, this.dtype)
    if (this.consistent) {
        this.data.copyInto(data, 0, 0, this.size)
    } else {
        var index = 0
        for (el in this)
            data[index++] = el
    }

    var index = this.size
    for (el in iter)
        data[index++] = el

    return data
}

/** Copies elements from [arrays] into [dest] along the given [axis]. Handles D1 through D4 explicitly. */
internal fun <T, D : Dimension, O : Dimension> concatenate(
    arrays: List<MultiArray<T, D>>, dest: NDArray<T, O>, axis: Int = 0
): NDArray<T, O> {
    if (axis == 0) {
        var offset = 0
        arrays.forEachIndexed { _: Int, arr: MultiArray<T, D> ->
            if (arr.consistent) {
                arr.data.copyInto(dest.data, offset, 0, arr.size)
            } else {
                var index = offset
                for (el in arr)
                    dest.data[index++] = el
            }
            offset += arr.size
        }
    } else {
        var index = 0
        val arrDim = arrays.first().dim
        for (i in 0 until dest.shape[0]) {
            when (arrDim) {
                D1 -> {
                    arrays as List<MultiArray<T, D1>>
                    for (array in arrays) {
                        dest.data[index++] = array[i]
                    }
                }

                D2 -> {
                    arrays as List<MultiArray<T, D2>>
                    when (axis) {
                        1 -> {
                            for (array in arrays) {
                                for (j in 0 until array.shape[1]) {
                                    dest.data[index++] = array[i, j]
                                }
                            }
                        }

                        2 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (array in arrays) {
                                    dest.data[index++] = array[i, j]
                                }
                            }
                        }
                    }
                }

                D3 -> {
                    arrays as List<MultiArray<T, D3>>
                    when (axis) {
                        1 -> {
                            for (array in arrays) {
                                for (j in 0 until array.shape[1]) {
                                    for (l in 0 until dest.shape[2]) {
                                        dest.data[index++] = array[i, j, l]
                                    }
                                }
                            }
                        }

                        2 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (array in arrays) {
                                    for (l in 0 until array.shape[2]) {
                                        dest.data[index++] = array[i, j, l]
                                    }
                                }
                            }
                        }

                        3 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (l in 0 until dest.shape[2]) {
                                    for (array in arrays) {
                                        dest.data[index++] = array[i, j, l]
                                    }
                                }
                            }
                        }
                    }
                }

                D4 -> {
                    arrays as List<MultiArray<T, D4>>
                    when (axis) {
                        1 -> {
                            for (array in arrays) {
                                for (j in 0 until array.shape[1]) {
                                    for (l in 0 until dest.shape[2]) {
                                        for (h in 0 until dest.shape[3]) {
                                            dest.data[index++] = array[i, j, l, h]
                                        }
                                    }
                                }
                            }
                        }

                        2 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (array in arrays) {
                                    for (l in 0 until array.shape[2]) {
                                        for (h in 0 until dest.shape[3]) {
                                            dest.data[index++] = array[i, j, l, h]
                                        }
                                    }
                                }
                            }
                        }

                        3 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (l in 0 until dest.shape[2]) {
                                    for (array in arrays) {
                                        for (h in 0 until array.shape[3]) {
                                            dest.data[index++] = array[i, j, l, h]
                                        }
                                    }
                                }
                            }
                        }

                        4 -> {
                            for (j in 0 until dest.shape[1]) {
                                for (l in 0 until dest.shape[2]) {
                                    for (h in 0 until dest.shape[3]) {
                                        for (array in arrays) {
                                            dest.data[index++] = array[i, j, l, h]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                else -> throw UnsupportedOperationException()
            }
        }
    }
    return dest
}

/**
 * Clamps every element to the range [[min], [max]], returning a new array.
 *
 * Elements below [min] become [min]; elements above [max] become [max].
 * The original array is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 5, 10])
 * a.clip(2, 8) // [2, 5, 8]
 * ```
 *
 * @param min the lower bound.
 * @param max the upper bound.
 * @return a new [NDArray] with all elements in `min..max`.
 * @throws IllegalArgumentException if [min] > [max].
 */
public fun <T, D : Dimension> MultiArray<T, D>.clip(min: T, max: T): NDArray<T, D> where T : Comparable<T>, T : Number {
    require(min <= max) {
        "min value for clipping should be lower than or equal to the [max] value"
    }
    val clippedData = initMemoryView(size, dtype) { index ->
        val e = data[index]
        if (e < min) min else if (e > max) max else e
    }
    return NDArray(data = clippedData, shape = shape.copyOf(), dim = dim)
}

/**
 * Inserts a size-1 axis at position [axis], promoting a 1D array to 2D.
 *
 * Returns a view when the array is [consistent][MultiArray.consistent], otherwise copies.
 *
 * @param axis position where the new axis is inserted.
 * @return a 2D view (or copy) with a size-1 dimension at [axis].
 * @see [unsqueeze]
 */
@JvmName("expandDimsD1")
public fun <T> MultiArray<T, D1>.expandDims(axis: Int): MultiArray<T, D2> {
    val newShape = shape.toMutableList().apply { add(axis, 1) }.toIntArray()
    // TODO(get rid of copying)
    val newData = if (consistent) this.data else this.deepCopy().data
    val newBase = if (consistent) this.base ?: this else null
    val newOffset = if (consistent) this.offset else 0
    return D2Array(newData, newOffset, newShape, dim = D2, base = newBase)
}

/** Inserts a size-1 axis at position [axis], promoting a 2D array to 3D. */
@JvmName("expandDimsD2")
public fun <T> MultiArray<T, D2>.expandDims(axis: Int): MultiArray<T, D3> {
    val newShape = shape.toMutableList().apply { add(axis, 1) }.toIntArray()
    // TODO(get rid of copying)
    val newData = if (consistent) this.data else this.deepCopy().data
    val newBase = if (consistent) this.base ?: this else null
    val newOffset = if (consistent) this.offset else 0
    return D3Array(newData, newOffset, newShape, dim = D3, base = newBase)
}

/** Inserts a size-1 axis at position [axis], promoting a 3D array to 4D. */
@JvmName("expandDimsD3")
public fun <T> MultiArray<T, D3>.expandDims(axis: Int): MultiArray<T, D4> {
    val newShape = shape.toMutableList().apply { add(axis, 1) }.toIntArray()
    // TODO(get rid of copying)
    val newData = if (consistent) this.data else this.deepCopy().data
    val newBase = if (consistent) this.base ?: this else null
    val newOffset = if (consistent) this.offset else 0
    return D4Array(newData, newOffset, newShape, dim = D4, base = newBase)
}

/** Inserts a size-1 axis at position [axis], promoting a 4D array to N-dimensional. */
@JvmName("expandDimsD4")
public fun <T> MultiArray<T, D4>.expandDims(axis: Int): MultiArray<T, DN> = this.unsqueeze()

/**
 * Inserts size-1 axes at the given positions, returning an N-dimensional array.
 *
 * @param axes positions where new axes are inserted.
 * @see [unsqueeze]
 */
@JvmName("expandDimsDN")
public fun <T, D : Dimension> MultiArray<T, D>.expandNDims(vararg axes: Int): MultiArray<T, DN> = this.unsqueeze()
