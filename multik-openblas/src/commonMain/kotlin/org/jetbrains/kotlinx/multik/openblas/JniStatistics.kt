/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas

import org.jetbrains.kotlinx.multik.api.Statistics
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object NativeStatistics : Statistics {

    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4> {
        TODO("Not yet implemented")
    }
}