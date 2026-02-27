package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Applies the cosine function element-wise to a numeric array, returning [Double] results.
 *
 * @param a the input array (values in radians).
 * @return a new [NDArray] where each element is `cos(a[i])`.
 * @see [MultiArray.cos] for the extension equivalent: `a.cos()`.
 */
@JvmName("cos")
public fun <T : Number, D : Dimension> Math.cos(a: MultiArray<T, D>): NDArray<Double, D> = this.mathEx.cos(a)

/**
 * Applies the cosine function element-wise to a float array, preserving single precision.
 *
 * @param a the input float array (values in radians).
 * @return a new [NDArray] where each element is `cos(a[i])`.
 */
@JvmName("cosFloat")
public fun <D : Dimension> Math.cos(a: MultiArray<Float, D>): NDArray<Float, D> = this.mathEx.cosF(a)

/**
 * Applies the cosine function element-wise to a [ComplexFloat] array.
 *
 * @param a the input complex float array.
 * @return a new [NDArray] where each element is `cos(a[i])`.
 */
@JvmName("cosComplexFloat")
public fun <D : Dimension> Math.cos(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> = this.mathEx.cosCF(a)

/**
 * Applies the cosine function element-wise to a [ComplexDouble] array.
 *
 * @param a the input complex double array.
 * @return a new [NDArray] where each element is `cos(a[i])`.
 */
@JvmName("cosComplexDouble")
public fun <D : Dimension> Math.cos(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> = this.mathEx.cosCD(a)
