/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.Math
import org.jetbrains.kotlinx.multik.jni.NativeMath
import org.jetbrains.kotlinx.multik.jvm.JvmMath
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object DefaultMath : Math {
    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        JvmMath.argMax(a)
    } else {
        NativeMath.argMax(a)
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        JvmMath.argMax(a, axis)

    override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMax(a, axis)

    override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMax(a, axis)

    override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMax(a, axis)

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMax(a, axis)

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        JvmMath.argMin(a)
    } else {
        NativeMath.argMin(a)
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        JvmMath.argMin(a, axis)

    override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMin(a, axis)

    override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMin(a, axis)

    override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMin(a, axis)

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMin(a, axis)

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> = NativeMath.exp(a)

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> = NativeMath.log(a)

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> = NativeMath.sin(a)

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> = NativeMath.cos(a)

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T = if (a.size <= 100) {
        JvmMath.max(a)
    } else {
        NativeMath.max(a)
    }

    override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        JvmMath.max(a, axis)

    override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = max(a, axis)

    override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = max(a, axis)

    override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = max(a, axis)

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = max(a, axis)

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T = if (a.size <= 100) {
        JvmMath.min(a)
    } else {
        NativeMath.min(a)
    }

    override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        JvmMath.min(a, axis)

    override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = min(a, axis)

    override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = min(a, axis)

    override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = min(a, axis)

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = min(a, axis)

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T = if (a.size <= 100) {
        JvmMath.sum(a)
    } else {
        NativeMath.sum(a)
    }

    override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        JvmMath.sum(a, axis)

    override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = sum(a, axis)

    override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = sum(a, axis)

    override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = sum(a, axis)

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = sum(a, axis)

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> = NativeMath.cumSum(a)

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> =
        JvmMath.cumSum(a, axis)

}