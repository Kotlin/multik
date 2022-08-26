/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.kotlin.stat

import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.kotlin.math.KEMath
import org.jetbrains.kotlinx.multik.kotlin.math.remove
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.jetbrains.kotlinx.multik.ndarray.operations.sorted
import org.jetbrains.kotlinx.multik.ndarray.operations.times

internal object KEStatistics : Statistics {
    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? {
        val size = a.size
        return when {
            size == 1 -> a.first().toDouble()
            size > 1 -> {
                val sorted = a.sorted()
                val mid = size / 2
                if (size % 2 != 0) {
                    sorted.data[mid].toDouble()
                } else {
                    (sorted.data[mid - 1].toDouble() + sorted.data[mid].toDouble()) / 2
                }
            }
            else -> null
        }
    }

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double {
        if (weights == null) return mean(a)
        return KEMath.sum(a * weights).toDouble() / KEMath.sum(weights).toDouble()
    }

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double {
        val ret = KEMath.sum(a)
        return ret.toDouble() / a.size
    }

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `mean` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val retData = initMemoryView<Double>(newShape.fold(1, Int::times), DataType.DoubleDataType)
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r until a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            val t = a.slice<T, D, O>(indexMap)
            var count = 0
            for (element in t) {
                retData[count] += element.toDouble()
                count++
            }
        }

        return NDArray<Double, O>(
            retData, 0, newShape, dim = dimensionOf(newShape.size)
        ) / a.shape[axis].toDouble()
    }

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1> = mean(a, axis)

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2> = mean(a, axis)

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3> = mean(a, axis)

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4> = mean(a, axis)
}
