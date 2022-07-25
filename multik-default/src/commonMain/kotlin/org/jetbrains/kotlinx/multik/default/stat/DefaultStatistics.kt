/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.stat

import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.ndarray.data.*

public expect object DefaultStatistics : Statistics {

    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double?

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O>

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1>

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2>

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3>

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4>
}