/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.stat

import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.kotlin.stat.KEStatistics
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultStatistics : Statistics {

    actual override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? = KEStatistics.median(a)

    actual override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double =
        KEStatistics.average(a, weights)

    actual override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double = KEStatistics.mean(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O> =
        KEStatistics.mean(a, axis)

    actual override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1> = mean(a, axis)

    actual override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2> = mean(a, axis)

    actual override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3> = mean(a, axis)

    actual override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4> = mean(a, axis)
}