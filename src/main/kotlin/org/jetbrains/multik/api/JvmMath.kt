package org.jetbrains.multik.api

import org.jetbrains.multik.core.*

object JvmMath : Math {
    override fun <T : Number, D : DN> argMax(a: Ndarray<T, D>): Int {
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

    override fun <T : Number, D : DN> argMin(a: Ndarray<T, D>): Int {
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

    override fun <T : Number, D : DN> exp(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.exp(it) }
    }

    override fun <T : Number, D : DN> log(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.ln(it) }
    }

    override fun <T : Number, D : DN> sin(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.sin(it) }
    }

    override fun <T : Number, D : DN> cos(a: Ndarray<T, D>): Ndarray<Double, D> {
        return mathOperation(a) { kotlin.math.cos(it) }
    }

    override fun <T : Number, D : DN> max(a: Ndarray<T, D>): T {
        var max = a.first()
        for (el in a) if (max < el) max = el
        return max
    }

    override fun <T : Number, D : DN> min(a: Ndarray<T, D>): T {
        var min = a.first()
        for (el in a) if (min > el) min = el
        return min
    }

    override fun <T : Number, D : DN> sum(a: Ndarray<T, D>): T {
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

    override fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>): D1Array<T> {
        val ret = D1Array<Double>(initMemoryView(a.size, DataType.DoubleDataType), shape = intArrayOf(a.size), dtype = DataType.DoubleDataType)
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
        return ret.asType<T>(a.dtype).toD1Array()
    }

    override fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>, axis: Int): Ndarray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : DN> mathOperation(a: Ndarray<T, D>, function: (Double) -> Double): Ndarray<Double, D> {
        val iter = a.iterator()
        val data = initMemoryView<Double>(a.size, DataType.DoubleDataType) {
            if (iter.hasNext())
                function(iter.next().toDouble())
            else
                0.0
        }
        return initNdarray(data, 0, a.shape, dtype = DataType.DoubleDataType, dim = a.dim)
    }
}