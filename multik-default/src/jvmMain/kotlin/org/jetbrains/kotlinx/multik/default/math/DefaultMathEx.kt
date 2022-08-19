/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public actual object DefaultMathEx : MathEx {

    private val natMathEx = DefaultEngineFactory.getEngine(NativeEngineType).getMath().mathEx

    actual override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> = natMathEx.exp(a)
    actual override fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D> = natMathEx.expF(a)
    actual override fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = natMathEx.expCF(a)
    actual override fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = natMathEx.expCD(a)

    actual override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> = natMathEx.log(a)
    actual override fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D> = natMathEx.logF(a)
    actual override fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = natMathEx.logCF(a)
    actual override fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = natMathEx.logCD(a)

    actual override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> = natMathEx.sin(a)
    actual override fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D> = natMathEx.sinF(a)
    actual override fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = natMathEx.sinCF(a)
    actual override fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = natMathEx.sinCD(a)

    actual override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> = natMathEx.cos(a)
    actual override fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D> = natMathEx.cosF(a)
    actual override fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = natMathEx.cosCF(a)
    actual override fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = natMathEx.cosCD(a)
}