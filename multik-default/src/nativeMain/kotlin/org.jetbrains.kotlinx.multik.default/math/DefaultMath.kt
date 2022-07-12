/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.jni.math.NativeMath
import org.jetbrains.kotlinx.multik.kotlin.math.KEMath
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultMath : Math {

    actual override val mathEx: MathEx
        get() = DefaultMathEx

    actual override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        KEMath.argMax(a)
    } else {
        NativeMath.argMax(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        KEMath.argMax(a, axis)

    actual override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMax(a, axis)

    actual override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMax(a, axis)

    actual override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMax(a, axis)

    actual override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMax(a, axis)

    actual override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        KEMath.argMin(a)
    } else {
        NativeMath.argMin(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        KEMath.argMin(a, axis)

    actual override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMin(a, axis)

    actual override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMin(a, axis)

    actual override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMin(a, axis)

    actual override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> = argMin(a, axis)

    actual override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T = if (a.size <= 100) {
        KEMath.max(a)
    } else {
        NativeMath.max(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        KEMath.max(a, axis)

    actual override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = max(a, axis)

    actual override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = max(a, axis)

    actual override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = max(a, axis)

    actual override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = max(a, axis)

    actual override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T = if (a.size <= 100) {
        KEMath.min(a)
    } else {
        NativeMath.min(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        KEMath.min(a, axis)

    actual override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = min(a, axis)

    actual override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = min(a, axis)

    actual override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = min(a, axis)

    actual override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = min(a, axis)

    actual override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T = if (a.size <= 100) {
        KEMath.sum(a)
    } else {
        NativeMath.sum(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        KEMath.sum(a, axis)

    actual override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = sum(a, axis)

    actual override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = sum(a, axis)

    actual override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = sum(a, axis)

    actual override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> = sum(a, axis)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> = NativeMath.cumSum(a)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> =
        KEMath.cumSum(a, axis)
}