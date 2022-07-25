/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Returns flat index of maximum element in an ndarray.
 *
 * same as [Math.argMax]
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMax(): Int = mk.math.argMax(this)

/**
 * Returns flat index of minimum element in an ndarray.
 *
 * same as [Math.argMin]
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMin(): Int = mk.math.argMin(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which an exp function has been applied.
 *
 * same as [Math.exp]
 */
@JvmName("expTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.exp(): NDArray<Double, D> = mk.math.exp(this)
/**
 * Returns an ndarray of Float from the given ndarray to each element of which an exp function has been applied.
 *
 * same as [Math.exp]
 */
@JvmName("expFloatD")
public fun <D : Dimension> MultiArray<Float, D>.exp(): NDArray<Float, D> = mk.math.mathEx.expF(this)
/**
 * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which an exp function has been applied.
 *
 * same as [Math.exp]
 */
@JvmName("expComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.exp(): NDArray<ComplexFloat, D> = mk.math.mathEx.expCF(this)
/**
 * Returns an ndarray of [ComplexDouble] from the given ndarray to each element of which an exp function has been applied.
 *
 * same as [Math.exp]
 */
@JvmName("expComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.exp(): NDArray<ComplexDouble, D> = mk.math.mathEx.expCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a log function has been applied.
 *
 * same as [Math.log]
 */
@JvmName("logTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.log(): NDArray<Double, D> = mk.math.mathEx.log(this)
/**
 * Returns an ndarray of Float from the given ndarray to each element of which a log function has been applied.
 *
 * same as [Math.log]
 */
@JvmName("logFloatD")
public fun <D : Dimension> MultiArray<Float, D>.log(): NDArray<Float, D> = mk.math.mathEx.logF(this)
/**
 * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which a log function has been applied.
 *
 * same as [Math.log]
 */
@JvmName("logComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.log(): NDArray<ComplexFloat, D> = mk.math.mathEx.logCF(this)
/**
 * Returns an ndarray of [ComplexDouble] from the given ndarray to each element of which a log function has been applied.
 *
 * same as [Math.log]
 */
@JvmName("logComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.log(): NDArray<ComplexDouble, D> = mk.math.mathEx.logCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a sin function has been applied.
 *
 * same as [Math.sin]
 */
@JvmName("sinTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sin(): NDArray<Double, D> = mk.math.mathEx.sin(this)
/**
 * Returns an ndarray of Float from the given ndarray to each element of which a sin function has been applied.
 *
 * same as [Math.sin]
 */
@JvmName("sinFloatD")
public fun <D : Dimension> MultiArray<Float, D>.sin(): NDArray<Float, D> = mk.math.mathEx.sinF(this)
/**
 * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which a sin function has been applied.
 *
 * same as [Math.sin]
 */
@JvmName("sinComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.sin(): NDArray<ComplexFloat, D> = mk.math.mathEx.sinCF(this)
/**
 * Returns an ndarray of [ComplexDouble] from the given ndarray to each element of which a sin function has been applied.
 *
 * same as [Math.sin]
 */
@JvmName("sinComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.sin(): NDArray<ComplexDouble, D> = mk.math.mathEx.sinCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a cos function has been applied.
 *
 * same as [Math.cos]
 */
@JvmName("cosTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.cos(): NDArray<Double, D> = mk.math.mathEx.cos(this)
/**
 * Returns an ndarray of Float from the given ndarray to each element of which a cos function has been applied.
 *
 * same as [Math.cos]
 */
@JvmName("cosFloatD")
public fun <D : Dimension> MultiArray<Float, D>.cos(): NDArray<Float, D> = mk.math.mathEx.cosF(this)
/**
 * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which a cos function has been applied.
 *
 * same as [Math.cos]
 */
@JvmName("cosComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.cos(): NDArray<ComplexFloat, D> = mk.math.mathEx.cosCF(this)
/**
 * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which a cos function has been applied.
 *
 * same as [Math.cos]
 */
@JvmName("cosComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.cos(): NDArray<ComplexDouble, D> = mk.math.mathEx.cosCD(this)

/**
 * Returns cumulative sum of all elements in the given ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.cumSum(): D1Array<T> = mk.math.cumSum(this)
