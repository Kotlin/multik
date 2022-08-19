/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.stat

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultStatistics : Statistics {

    private val ktStat = DefaultEngineFactory.getEngine(KEEngineType).getStatistics()
    private val natStat = DefaultEngineFactory.getEngine(NativeEngineType).getStatistics()

    actual override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? = natStat.median(a)

    actual override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double =
        ktStat.average(a, weights)

    actual override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double = ktStat.mean(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O> =
        ktStat.mean(a, axis)

    actual override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1> = mean(a, axis)

    actual override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2> = mean(a, axis)

    actual override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3> = mean(a, axis)

    actual override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4> = mean(a, axis)
}