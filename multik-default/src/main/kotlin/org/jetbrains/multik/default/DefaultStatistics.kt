package org.jetbrains.multik.default

import org.jetbrains.multik.api.Statistics
import org.jetbrains.multik.jvm.JvmStatistics
import org.jetbrains.multik.ndarray.data.*

public object DefaultStatistics : Statistics {
    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? = JvmStatistics.median(a)

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double =
        JvmStatistics.average(a, weights)

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double = JvmStatistics.mean(a)

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): Ndarray<Double, O> =
        JvmStatistics.mean(a, axis)

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): Ndarray<Double, D1> = mean(a, axis)

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): Ndarray<Double, D2> = mean(a, axis)

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): Ndarray<Double, D3> = mean(a, axis)

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): Ndarray<Double, D4> = mean(a, axis)

}