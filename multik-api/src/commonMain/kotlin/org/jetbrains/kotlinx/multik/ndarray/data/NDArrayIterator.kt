/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Iterator over multidimensional arrays. Iterated taking into account the [offset], [strides] and [shape].
 */
public class NDArrayIterator<T : Number>(
    private val data: MemoryView<T>,
    private val offset: Int = 0,
    private val strides: IntArray,
    private val shape: IntArray
) : Iterator<T> {
    private val index = IntArray(shape.size)

    override fun hasNext(): Boolean {
        for (i in shape.indices) {
            if (index[i] >= shape[i])
                return false
        }
        return true
    }

    override fun next(): T {
        var p = offset
        for (i in shape.indices) {
            p += strides[i] * index[i]
        }

        for (i in shape.size - 1 downTo 0) {
            val t = index[i] + 1
            if (t >= shape[i] && i != 0) {
                index[i] = 0
            } else {
                index[i] = t
                break
            }
        }

        return data[p]
    }
}