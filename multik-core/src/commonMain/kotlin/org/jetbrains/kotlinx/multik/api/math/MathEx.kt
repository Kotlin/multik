/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Extension interface for [Math] for improved type support.
 */
public interface MathEx {
    public fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D>
    public fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D>
    public fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    public fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    public fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D>
    public fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D>
    public fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    public fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    public fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D>
    public fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D>
    public fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    public fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    public fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D>
    public fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D>
    public fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    public fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>
}