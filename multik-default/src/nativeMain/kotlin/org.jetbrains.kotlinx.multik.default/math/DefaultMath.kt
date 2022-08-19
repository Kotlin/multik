/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultMath : Math {

    private val ktMath = DefaultEngineFactory.getEngine(KEEngineType).getMath()
    private val natMath = DefaultEngineFactory.getEngine(NativeEngineType).getMath()

    actual override val mathEx: MathEx
        get() = DefaultMathEx

    actual override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        ktMath.argMax(a)
    } else {
        natMath.argMax(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        ktMath.argMax(a, axis)

    actual override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMax(a, axis)

    actual override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMax(a, axis)

    actual override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMax(a, axis)

    actual override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> = argMax(a, axis)

    actual override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int = if (a.size <= 100) {
        ktMath.argMin(a)
    } else {
        natMath.argMin(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        ktMath.argMin(a, axis)

    actual override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMin(a, axis)

    actual override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMin(a, axis)

    actual override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMin(a, axis)

    actual override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> = argMin(a, axis)

    actual override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T = if (a.size <= 100) {
        ktMath.max(a)
    } else {
        natMath.max(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.max(a, axis)

    actual override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = max(a, axis)

    actual override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = max(a, axis)

    actual override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = max(a, axis)

    actual override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = max(a, axis)

    actual override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T = if (a.size <= 100) {
        ktMath.min(a)
    } else {
        natMath.min(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.min(a, axis)

    actual override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = min(a, axis)

    actual override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = min(a, axis)

    actual override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = min(a, axis)

    actual override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = min(a, axis)

    actual override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T = if (a.size <= 100) {
        ktMath.sum(a)
    } else {
        natMath.sum(a)
    }

    actual override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.sum(a, axis)

    actual override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = sum(a, axis)

    actual override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = sum(a, axis)

    actual override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = sum(a, axis)

    actual override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = sum(a, axis)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> = natMath.cumSum(a)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> =
        ktMath.cumSum(a, axis)
}