package org.jetbrains.multik.jvm

import org.jetbrains.multik.api.Math
import org.jetbrains.multik.ndarray.data.*
import org.jetbrains.multik.ndarray.operations.first
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

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.exp(it) }
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { ln(it) }
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.sin(it) }
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.cos(it) }
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        var max = a.first()
        for (el in a) if (max < el) max = el
        return max
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        var min = a.first()
        for (el in a) if (min > el) min = el
        return min
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        var accum = 0.0
        var compens = 0.0 // compensation
        for (el in a) {
            val y = el.toDouble() - compens
            val t = accum + y
            compens = t - accum - y
            accum = t
        }
        return accum.toPrimitiveType(a.dtype)
    }

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
        return ret.asType<T>(a.dtype)
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): Ndarray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : Dimension> mathOperation(
        a: MultiArray<T, D>, function: (Double) -> Double
    ): Ndarray<Double, D> {
        val iter = a.iterator()
        val data = initMemoryView<Double>(a.size, DataType.DoubleDataType) {
            if (iter.hasNext())
                function(iter.next().toDouble())
            else
                0.0
        }
        return Ndarray<Double, D>(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }
}