/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Returns a ndarray of Double from the given ndarray to each element of which a log function has been applied.
 */
@JvmName("log")
public fun <T : Number, D : Dimension> Math.log(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.log(a)

@JvmName("logFloat")
public fun <D : Dimension> Math.log(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.logF(a)

@JvmName("logComplexFloat")
public fun <D : Dimension> Math.log(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.logCF(a)

@JvmName("logComplexDouble")
public fun <D : Dimension> Math.log(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.logCD(a)