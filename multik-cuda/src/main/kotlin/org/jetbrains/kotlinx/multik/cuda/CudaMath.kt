/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import org.jetbrains.kotlinx.multik.api.Math
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object CudaMath : Math {
    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> {
        TODO("Not yet implemented")
    }

}