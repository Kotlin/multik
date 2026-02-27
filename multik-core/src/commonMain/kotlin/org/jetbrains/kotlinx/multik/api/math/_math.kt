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
 * Returns the flat index of the maximum element in this array.
 *
 * @return the flat index of the maximum element.
 * @see [Math.argMax] for the functional-style equivalent via `mk.math.argMax(a)`.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMax(): Int = mk.math.argMax(this)

/**
 * Returns the flat index of the minimum element in this array.
 *
 * @return the flat index of the minimum element.
 * @see [Math.argMin] for the functional-style equivalent via `mk.math.argMin(a)`.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMin(): Int = mk.math.argMin(this)

// ---- exp ----

/**
 * Applies the exponential function element-wise, returning [Double] results.
 *
 * @return a new [NDArray] where each element is `e^(this[i])`.
 * @see [Math.exp] for the functional-style equivalent via `mk.math.exp(a)`.
 */
@JvmName("expTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.exp(): NDArray<Double, D> = mk.math.exp(this)

/** Applies the exponential function element-wise to a float array, preserving single precision. */
@JvmName("expFloatD")
public fun <D : Dimension> MultiArray<Float, D>.exp(): NDArray<Float, D> = mk.math.mathEx.expF(this)

/** Applies the exponential function element-wise to a [ComplexFloat] array. */
@JvmName("expComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.exp(): NDArray<ComplexFloat, D> = mk.math.mathEx.expCF(this)

/** Applies the exponential function element-wise to a [ComplexDouble] array. */
@JvmName("expComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.exp(): NDArray<ComplexDouble, D> = mk.math.mathEx.expCD(this)

// ---- log ----

/**
 * Applies the natural logarithm element-wise, returning [Double] results.
 *
 * @return a new [NDArray] where each element is `ln(this[i])`.
 * @see [Math.log] for the functional-style equivalent via `mk.math.log(a)`.
 */
@JvmName("logTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.log(): NDArray<Double, D> = mk.math.mathEx.log(this)

/** Applies the natural logarithm element-wise to a float array, preserving single precision. */
@JvmName("logFloatD")
public fun <D : Dimension> MultiArray<Float, D>.log(): NDArray<Float, D> = mk.math.mathEx.logF(this)

/** Applies the natural logarithm element-wise to a [ComplexFloat] array. */
@JvmName("logComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.log(): NDArray<ComplexFloat, D> = mk.math.mathEx.logCF(this)

/** Applies the natural logarithm element-wise to a [ComplexDouble] array. */
@JvmName("logComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.log(): NDArray<ComplexDouble, D> = mk.math.mathEx.logCD(this)

// ---- sin ----

/**
 * Applies the sine function element-wise, returning [Double] results.
 *
 * @return a new [NDArray] where each element is `sin(this[i])`.
 * @see [Math.sin] for the functional-style equivalent via `mk.math.sin(a)`.
 */
@JvmName("sinTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sin(): NDArray<Double, D> = mk.math.mathEx.sin(this)

/** Applies the sine function element-wise to a float array, preserving single precision. */
@JvmName("sinFloatD")
public fun <D : Dimension> MultiArray<Float, D>.sin(): NDArray<Float, D> = mk.math.mathEx.sinF(this)

/** Applies the sine function element-wise to a [ComplexFloat] array. */
@JvmName("sinComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.sin(): NDArray<ComplexFloat, D> = mk.math.mathEx.sinCF(this)

/** Applies the sine function element-wise to a [ComplexDouble] array. */
@JvmName("sinComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.sin(): NDArray<ComplexDouble, D> = mk.math.mathEx.sinCD(this)

// ---- cos ----

/**
 * Applies the cosine function element-wise, returning [Double] results.
 *
 * @return a new [NDArray] where each element is `cos(this[i])`.
 * @see [Math.cos] for the functional-style equivalent via `mk.math.cos(a)`.
 */
@JvmName("cosTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.cos(): NDArray<Double, D> = mk.math.mathEx.cos(this)

/** Applies the cosine function element-wise to a float array, preserving single precision. */
@JvmName("cosFloatD")
public fun <D : Dimension> MultiArray<Float, D>.cos(): NDArray<Float, D> = mk.math.mathEx.cosF(this)

/** Applies the cosine function element-wise to a [ComplexFloat] array. */
@JvmName("cosComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.cos(): NDArray<ComplexFloat, D> = mk.math.mathEx.cosCF(this)

/** Applies the cosine function element-wise to a [ComplexDouble] array. */
@JvmName("cosComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.cos(): NDArray<ComplexDouble, D> = mk.math.mathEx.cosCD(this)

// ---- cumSum ----

/**
 * Returns the cumulative sum of all elements as a flattened 1D array.
 *
 * @return a [D1Array] where each element is the running total.
 * @see [Math.cumSum] for the functional-style equivalent via `mk.math.cumSum(a)`.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.cumSum(): D1Array<T> = mk.math.cumSum(this)
