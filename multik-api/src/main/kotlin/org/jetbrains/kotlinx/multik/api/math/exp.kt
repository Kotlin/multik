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
 * Returns an ndarray of Double from the given ndarray to each element of which an exp function has been applied.
 */
@JvmName("exp")
public fun <T : Number, D : Dimension> Math.exp(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.exp(a)

@JvmName("expFloat")
public fun <D : Dimension> Math.exp(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.expF(a)

@JvmName("expComplexFloat")
public fun <D : Dimension> Math.exp(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.expCF(a)

@JvmName("expComplexDouble")
public fun <D : Dimension> Math.exp(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.expCD(a)