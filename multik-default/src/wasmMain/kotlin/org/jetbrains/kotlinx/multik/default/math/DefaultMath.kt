/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultMath : Math {

    private val ktMath = DefaultEngineFactory.getEngine(KEEngineType).getMath()

    actual override val mathEx: MathEx
        get() = DefaultMathEx

    actual override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int = ktMath.argMax(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        ktMath.argMax(a, axis)

    actual override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMax(a, axis)

    actual override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMax(a, axis)

    actual override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMax(a, axis)

    actual override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> = argMax(a, axis)

    actual override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int = ktMath.argMin(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> =
        ktMath.argMin(a, axis)

    actual override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> = argMin(a, axis)

    actual override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> = argMin(a, axis)

    actual override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> = argMin(a, axis)

    actual override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, DN> = argMin(a, axis)

    actual override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T = ktMath.max(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.max(a, axis)

    actual override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = max(a, axis)

    actual override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = max(a, axis)

    actual override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = max(a, axis)

    actual override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = max(a, axis)

    actual override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T = ktMath.min(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.min(a, axis)

    actual override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = min(a, axis)

    actual override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = min(a, axis)

    actual override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = min(a, axis)

    actual override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = min(a, axis)

    actual override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T = ktMath.sum(a)

    actual override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> =
        ktMath.sum(a, axis)

    actual override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> = sum(a, axis)

    actual override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> = sum(a, axis)

    actual override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> = sum(a, axis)

    actual override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, DN> = sum(a, axis)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> = ktMath.cumSum(a)

    actual override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> =
        ktMath.cumSum(a, axis)
}