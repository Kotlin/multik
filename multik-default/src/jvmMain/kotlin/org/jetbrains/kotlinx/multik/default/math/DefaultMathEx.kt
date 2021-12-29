/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.jni.NativeEngine
import org.jetbrains.kotlinx.multik.jni.math.NativeMathEx
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object DefaultMathEx : MathEx {

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> = NativeMathEx.exp(a)
    override fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D> = NativeMathEx.expF(a)
    override fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = NativeMathEx.expCF(a)
    override fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        NativeMathEx.expCD(a)

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> = NativeMathEx.log(a)
    override fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D> = NativeMathEx.logF(a)
    override fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = NativeMathEx.logCF(a)
    override fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        NativeMathEx.logCD(a)

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> = NativeMathEx.sin(a)
    override fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D> = NativeMathEx.sinF(a)
    override fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = NativeMathEx.sinCF(a)
    override fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        NativeMathEx.sinCD(a)

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> = NativeMathEx.cos(a)
    override fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D> = NativeMathEx.cosF(a)
    override fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = NativeMathEx.cosCF(a)
    override fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        NativeMathEx.cosCD(a)
}