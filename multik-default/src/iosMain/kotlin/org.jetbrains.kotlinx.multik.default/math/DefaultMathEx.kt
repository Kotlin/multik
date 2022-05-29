/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.jvm.math.JvmMathEx
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public actual object DefaultMathEx : MathEx {

    actual override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> = JvmMathEx.exp(a)
    actual override fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D> = JvmMathEx.expF(a)
    actual override fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = JvmMathEx.expCF(a)
    actual override fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = JvmMathEx.expCD(a)

    actual override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> = JvmMathEx.log(a)
    actual override fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D> = JvmMathEx.logF(a)
    actual override fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = JvmMathEx.logCF(a)
    actual override fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = JvmMathEx.logCD(a)

    actual override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> = JvmMathEx.sin(a)
    actual override fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D> = JvmMathEx.sinF(a)
    actual override fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = JvmMathEx.sinCF(a)
    actual override fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = JvmMathEx.sinCD(a)

    actual override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> = JvmMathEx.cos(a)
    actual override fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D> = JvmMathEx.cosF(a)
    actual override fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = JvmMathEx.cosCF(a)
    actual override fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = JvmMathEx.cosCD(a)
}