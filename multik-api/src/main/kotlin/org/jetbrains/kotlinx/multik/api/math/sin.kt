/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a sin function has been applied.
 */
@JvmName("sin")
public fun <T : Number, D : Dimension> Math.sin(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.sin(a)

@JvmName("sinFloat")
public fun <D : Dimension> Math.sin(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.sinF(a)

@JvmName("sinComplexFloat")
public fun <D : Dimension> Math.sin(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.sinCF(a)

@JvmName("sinComplexDouble")
public fun <D : Dimension> Math.sin(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.sinCD(a)