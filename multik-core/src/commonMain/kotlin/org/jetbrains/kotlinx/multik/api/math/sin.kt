package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Applies the sine function element-wise to a numeric array, returning [Double] results.
 *
 * @param a the input array (values in radians).
 * @return a new [NDArray] where each element is `sin(a[i])`.
 * @see [MultiArray.sin] for the extension equivalent: `a.sin()`.
 */
@JvmName("sin")
public fun <T : Number, D : Dimension> Math.sin(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.sin(a)

/**
 * Applies the sine function element-wise to a float array, preserving single precision.
 *
 * @param a the input float array (values in radians).
 * @return a new [NDArray] where each element is `sin(a[i])`.
 */
@JvmName("sinFloat")
public fun <D : Dimension> Math.sin(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.sinF(a)

/**
 * Applies the sine function element-wise to a [ComplexFloat] array.
 *
 * @param a the input complex float array.
 * @return a new [NDArray] where each element is `sin(a[i])`.
 */
@JvmName("sinComplexFloat")
public fun <D : Dimension> Math.sin(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.sinCF(a)

/**
 * Applies the sine function element-wise to a [ComplexDouble] array.
 *
 * @param a the input complex double array.
 * @return a new [NDArray] where each element is `sin(a[i])`.
 */
@JvmName("sinComplexDouble")
public fun <D : Dimension> Math.sin(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.sinCD(a)
