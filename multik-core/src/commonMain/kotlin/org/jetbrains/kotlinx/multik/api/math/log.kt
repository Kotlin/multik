package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Applies the natural logarithm element-wise to a numeric array, returning [Double] results.
 *
 * @param a the input array.
 * @return a new [NDArray] where each element is `ln(a[i])`.
 * @see [MultiArray.log] for the extension equivalent: `a.log()`.
 */
@JvmName("log")
public fun <T : Number, D : Dimension> Math.log(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.log(a)

/**
 * Applies the natural logarithm element-wise to a float array, preserving single precision.
 *
 * @param a the input float array.
 * @return a new [NDArray] where each element is `ln(a[i])`.
 */
@JvmName("logFloat")
public fun <D : Dimension> Math.log(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.logF(a)

/**
 * Applies the natural logarithm element-wise to a [ComplexFloat] array.
 *
 * @param a the input complex float array.
 * @return a new [NDArray] where each element is `ln(a[i])`.
 */
@JvmName("logComplexFloat")
public fun <D : Dimension> Math.log(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.logCF(a)

/**
 * Applies the natural logarithm element-wise to a [ComplexDouble] array.
 *
 * @param a the input complex double array.
 * @return a new [NDArray] where each element is `ln(a[i])`.
 */
@JvmName("logComplexDouble")
public fun <D : Dimension> Math.log(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.logCD(a)
