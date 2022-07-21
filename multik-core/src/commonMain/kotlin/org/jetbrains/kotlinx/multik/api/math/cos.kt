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
 * Returns a ndarray of Double from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cos")
public fun <T : Number, D : Dimension> Math.cos(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.cos(a)

/**
 * Returns a ndarray of Float from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cosFloat")
public fun <D : Dimension> Math.cos(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.cosF(a)

/**
 * Returns a ndarray of [ComplexFloat] from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cosComplexFloat")
public fun <D : Dimension> Math.cos(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.cosCF(a)

/**
 * Returns a ndarray of [ComplexDouble] from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cosComplexDouble")
public fun <D : Dimension> Math.cos(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.cosCD(a)