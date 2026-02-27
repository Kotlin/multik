package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Applies the exponential function element-wise to a numeric array, returning [Double] results.
 *
 * @param a the input array.
 * @return a new [NDArray] where each element is `e^(a[i])`.
 * @see [MultiArray.exp] for the extension equivalent: `a.exp()`.
 */
@JvmName("exp")
public fun <T : Number, D : Dimension> Math.exp(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.exp(a)

/**
 * Applies the exponential function element-wise to a float array, preserving single precision.
 *
 * @param a the input float array.
 * @return a new [NDArray] where each element is `e^(a[i])`.
 */
@JvmName("expFloat")
public fun <D : Dimension> Math.exp(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.expF(a)

/**
 * Applies the exponential function element-wise to a [ComplexFloat] array.
 *
 * @param a the input complex float array.
 * @return a new [NDArray] where each element is `e^(a[i])`.
 */
@JvmName("expComplexFloat")
public fun <D : Dimension> Math.exp(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.expCF(a)

/**
 * Applies the exponential function element-wise to a [ComplexDouble] array.
 *
 * @param a the input complex double array.
 * @return a new [NDArray] where each element is `e^(a[i])`.
 */
@JvmName("expComplexDouble")
public fun <D : Dimension> Math.exp(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.expCD(a)
