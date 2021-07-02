/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.Math
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import kotlin.math.ln

public object JvmMath : Math {
    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        var arg = 0
        var count = 0
        var max = a.first()
        for (el in a) {
            if (max < el) {
                max = el
                arg = count
            }
            count++
        }
        return arg
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `argMax` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val min = JvmMath.min(a)
        val size = newShape.fold(1, Int::times)
        val maxArray = MutableList(size) { min }
        val argMaxData = initMemoryView<Int>(size, DataType.IntDataType)
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            val t = a.slice<T, D, O>(indexMap)
            var count = 0
            for (el in t) {
                if (maxArray[count] < el) {
                    argMaxData[count] = index
                    maxArray[count] = el
                }
                count++
            }
        }
        return NDArray(argMaxData, 0, newShape, dtype = DataType.IntDataType, dim = dimensionOf(newShape.size))
    }

    override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMax(a, axis)

    override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMax(a, axis)

    override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMax(a, axis)

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMax(a, axis)

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        var arg = 0
        var count = 0
        var min = a.first()
        for (el in a) {
            if (min > el) {
                min = el
                arg = count
            }
            count++
        }
        return arg
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `argMin` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val max = JvmMath.max(a)
        val size = newShape.fold(1, Int::times)
        val minArray = MutableList(size) { max }
        val argMinData = initMemoryView<Int>(size, DataType.IntDataType)
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            val t = a.slice<T, D, O>(indexMap)
            var count = 0
            for (el in t) {
                if (minArray[count] > el) {
                    argMinData[count] = index
                    minArray[count] = el
                }
                count++
            }
        }
        return NDArray(argMinData, 0, newShape, dtype = DataType.IntDataType, dim = dimensionOf(newShape.size))
    }

    override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMin(a, axis)

    override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMin(a, axis)

    override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMin(a, axis)

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMin(a, axis)

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a) { kotlin.math.exp(it) }
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a) { ln(it) }
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a) { kotlin.math.sin(it) }
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> {
        return mathOperation(a) { kotlin.math.cos(it) }
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        var max = a.first()
        for (el in a) if (max < el) max = el
        return max
    }

    override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `max` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val min = JvmMath.min(a)
        val size = newShape.fold(1, Int::times)
        val maxData = initMemoryView(size, a.dtype) { min }
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            val t = a.slice<T, D, O>(indexMap)
            var count = 0
            for (el in t) {
                if (maxData[count] < el) {
                    maxData[count] = el
                }
                count++
            }
        }
        return NDArray(maxData, 0, newShape, dtype = a.dtype, dim = dimensionOf(newShape.size))
    }

    override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = max(a, axis)

    override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = max(a, axis)

    override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = max(a, axis)

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = max(a, axis)

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        var min = a.first()
        for (el in a) if (min > el) min = el
        return min
    }

    override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `min` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val max = JvmMath.max(a)
        val size = newShape.fold(1, Int::times)
        val minData = initMemoryView(size, a.dtype) { max }
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            val t = a.slice<T, D, O>(indexMap)
            var count = 0
            for (el in t) {
                if (minData[count] > el) {
                    minData[count] = el
                }
                count++
            }
        }
        return NDArray(minData, 0, newShape, dtype = a.dtype, dim = dimensionOf(newShape.size))
    }

    override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = min(a, axis)

    override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = min(a, axis)

    override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = min(a, axis)

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = min(a, axis)

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T = summation(a)

    override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        require(a.dim.d > 1) { "NDArray of dimension one, use the `sum` function without axis." }
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val newShape = a.shape.remove(axis)
        val ret: NDArray<T, O> = mk.empty(newShape, a.dtype)
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 0 until a.shape[axis]) {
            indexMap[axis] = index.r
            ret += a.slice(indexMap)
        }
        return ret
    }

    override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = JvmMath.sum(a, axis)

    override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = JvmMath.sum(a, axis)

    override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = JvmMath.sum(a, axis)

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = JvmMath.sum(a, axis)

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        val ret = D1Array<Double>(
            initMemoryView(a.size, DataType.DoubleDataType),
            shape = intArrayOf(a.size),
            dtype = DataType.DoubleDataType,
            dim = D1
        )
        var ind = 0
        var accum = 0.0
        var compens = 0.0 // compensation
        for (el in a) {
            val y = el.toDouble() - compens
            val t = accum + y
            compens = t - accum - y
            accum = t
            ret[ind++] = accum
        }
        return ret.asType(a.dtype)
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> {
        require(axis in 0 until a.dim.d) { "axis $axis is out of bounds for this ndarray of dimension ${a.dim.d}." }
        val ret: NDArray<T, D> = (a as NDArray<T, D>).deepCopy()
        val indexMap: MutableMap<Int, Indexing> = mutableMapOf()
        for (i in a.shape.indices) {
            if (i == axis) continue
            indexMap[i] = 0.r..a.shape[i]
        }
        for (index in 1 until a.shape[axis]) {
            indexMap[axis] = index.r
            val tmp = ret.slice<T, D, DN>(indexMap)
            indexMap[axis] = index.r - 1.r
            tmp += ret.slice(indexMap)
        }
        return ret
    }

    private fun <T : Number, D : Dimension> mathOperation(
        a: MultiArray<T, D>, function: (Double) -> Double
    ): NDArray<Double, D> {
        val iter = a.iterator()
        val data = initMemoryView(a.size, DataType.DoubleDataType) {
            if (iter.hasNext())
                function(iter.next().toDouble())
            else
                0.0
        }
        return NDArray(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }
}

//todo util module
internal fun IntArray.remove(pos: Int): IntArray = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
}

private fun <T : Number, D : Dimension> summation(a: MultiArray<T, D>): T = when (a.dtype) {
    DataType.FloatDataType -> {
        var accum = 0.0f
        var compens = 0.0f // compensation
        val iter = if (a.consistent) a.data.getFloatArray().iterator() else (a as MultiArray<Float, *>).iterator()
        for (el in iter) {
            val y = el - compens
            val t = accum + y
            compens = t - accum - y
            accum = t
        }
        accum
    }
    DataType.DoubleDataType -> {
        var accum = 0.0
        var compens = 0.0 // compensation
        val iter = if (a.consistent) a.data.getDoubleArray().iterator() else (a as MultiArray<Double, *>).iterator()
        for (el in iter) {
            val y = el - compens
            val t = accum + y
            compens = t - accum - y
            accum = t
        }
        accum
    }
    DataType.IntDataType -> {
        var accum = 0
        val iter = if (a.consistent) a.data.getIntArray().iterator() else (a as MultiArray<Int, *>).iterator()
        for (el in iter) {
            accum += el
        }
        accum
    }
    DataType.LongDataType -> {
        var accum = 0L
        val iter = if (a.consistent) a.data.getLongArray().iterator() else (a as MultiArray<Long, *>).iterator()
        for (el in iter) {
            accum += el
        }
        accum
    }
    DataType.ShortDataType -> {
        var accum = 0
        val iter = if (a.consistent) a.data.getShortArray().iterator() else (a as MultiArray<Short, *>).iterator()
        for (el in iter) {
            accum += el
        }
        accum
    }
    DataType.ByteDataType -> {
        var accum = 0
        val iter = if (a.consistent) a.data.getByteArray().iterator() else (a as MultiArray<Byte, *>).iterator()
        for (el in iter) {
            accum += el
        }
        accum
    }
} as T