/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.copyInto
import org.jetbrains.kotlinx.multik.ndarray.data.*

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
public fun <T, D : Dimension, ID : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, ID>): D1Array<T> {
    val newSize = this.size + arr.size
    val data = this.copyFromTwoArrays(arr.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, D>, axis: Int): NDArray<T, D> =
    this.cat(arr, axis)

internal fun <T, D : Dimension> MultiArray<T, D>.copyFromTwoArrays(iter: Iterator<T>, size: Int): MemoryView<T> {
    val data = initMemoryView<T>(size, this.dtype)
    if (this.consistent) {
        when (this.dtype) {
            DataType.ComplexFloatDataType -> this.data.getComplexFloatArray().copyInto(data.getComplexFloatArray())
            DataType.ComplexDoubleDataType -> this.data.getComplexDoubleArray().copyInto(data.getComplexDoubleArray())
            else -> System.arraycopy(this.data.data, 0, data.data, 0, this.size)
        }
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
