/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(vararg value: T): D1Array<T> {
    val newSize = this.size + value.size
    val data = this.copyFromTwoArrays(value.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 *
 */
public infix fun <T, D : Dimension, ID : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, ID>): D1Array<T> {
    val newSize = this.size + arr.size
    val data = this.copyFromTwoArrays(arr.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, D>, axis: Int): NDArray<T, D> =
    this.cat(arr, axis)

@JvmName("stackD1")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D1>, axis: Int = 0): NDArray<T, D2> =
    stack(arr.toList(), axis)

@JvmName("stackD1")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D1>>, axis: Int = 0): NDArray<T, D2> =
    stackArrays(arrays, axis)

@JvmName("stackD2")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D2>, axis: Int = 0): NDArray<T, D3> =
    stack(arr.toList(), axis)

@JvmName("stackD2")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D2>>, axis: Int = 0): NDArray<T, D3> =
    stackArrays(arrays, axis)

@JvmName("stackD3")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D3>, axis: Int = 0): NDArray<T, D4> =
    stack(arr.toList(), axis)

@JvmName("stackD3")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D3>>, axis: Int = 0): NDArray<T, D4> =
    stackArrays(arrays, axis)

@JvmName("stackD4")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D4>, axis: Int = 0): NDArray<T, DN> =
    stack(arr.toList(), axis)

@JvmName("stackD4")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D4>>, axis: Int = 0): NDArray<T, DN> =
    stackArrays(arrays, axis)

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
 *
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
 * Clips the values in ndarray if value is not in range min..max
 *
 * values bigger than [max] are set to [max]
 *
 * values smaller than [min] are set to [min]
 *
 * @param min minimum value for clipping where any value in ndarray
 * that is smaller than [min] are clipped and set to [min]
 * @param max maximum value for clipping where any value in ndarray
 * that is bigger than [max] are clipped and set to [max]
 * @return NDArray of which all of its elements are in range min..max
 * @throws IllegalArgumentException if min > max
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
 * Returns a ndarray with an expanded shape.
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

/**
 * Returns a ndarray with an expanded shape.
 */
@JvmName("expandDimsD2")
public fun <T> MultiArray<T, D2>.expandDims(axis: Int): MultiArray<T, D3> {
    val newShape = shape.toMutableList().apply { add(axis, 1) }.toIntArray()
    // TODO(get rid of copying)
    val newData = if (consistent) this.data else this.deepCopy().data
    val newBase = if (consistent) this.base ?: this else null
    val newOffset = if (consistent) this.offset else 0
    return D3Array(newData, newOffset, newShape, dim = D3, base = newBase)
}

/**
 * Returns a ndarray with an expanded shape.
 */
@JvmName("expandDimsD3")
public fun <T> MultiArray<T, D3>.expandDims(axis: Int): MultiArray<T, D4> {
    val newShape = shape.toMutableList().apply { add(axis, 1) }.toIntArray()
    // TODO(get rid of copying)
    val newData = if (consistent) this.data else this.deepCopy().data
    val newBase = if (consistent) this.base ?: this else null
    val newOffset = if (consistent) this.offset else 0
    return D4Array(newData, newOffset, newShape, dim = D4, base = newBase)
}

/**
 * Returns a ndarray with an expanded shape.
 */
@JvmName("expandDimsD4")
public fun <T> MultiArray<T, D4>.expandDims(axis: Int): MultiArray<T, DN> = this.unsqueeze()

/**
 * Returns a ndarray with an expanded shape.
 *
 * @see MultiArray.unsqueeze
 */
@JvmName("expandDimsDN")
public fun <T, D : Dimension> MultiArray<T, D>.expandNDims(vararg axes: Int): MultiArray<T, DN> = this.unsqueeze()
