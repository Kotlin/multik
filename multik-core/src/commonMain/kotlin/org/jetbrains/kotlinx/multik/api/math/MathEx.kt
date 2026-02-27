package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Extended math operations interface providing element-wise transcendental functions.
 *
 * Each function has four variants by element type: numeric (`Number` → `Double`),
 * float-specific (`Float` → `Float`), [ComplexFloat], and [ComplexDouble].
 * Numeric variants promote to `Double`; float variants preserve single precision.
 *
 * Users typically call the convenience extensions on [Math] (e.g., `mk.math.sin(a)`)
 * or the [MultiArray] extension (e.g., `a.sin()`) rather than accessing this interface directly.
 *
 * @see [Math] for the main entry point.
 */
public interface MathEx {

    // ---- exp ----

    /** Applies the exponential function element-wise to a numeric array, returning [Double] results. */
    public fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D>

    /** Applies the exponential function element-wise to a float array, preserving single precision. */
    public fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D>

    /** Applies the exponential function element-wise to a [ComplexFloat] array. */
    public fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>

    /** Applies the exponential function element-wise to a [ComplexDouble] array. */
    public fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    // ---- log ----

    /** Applies the natural logarithm element-wise to a numeric array, returning [Double] results. */
    public fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D>

    /** Applies the natural logarithm element-wise to a float array, preserving single precision. */
    public fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D>

    /** Applies the natural logarithm element-wise to a [ComplexFloat] array. */
    public fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>

    /** Applies the natural logarithm element-wise to a [ComplexDouble] array. */
    public fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    // ---- sin ----

    /** Applies the sine function element-wise to a numeric array, returning [Double] results. */
    public fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D>

    /** Applies the sine function element-wise to a float array, preserving single precision. */
    public fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D>

    /** Applies the sine function element-wise to a [ComplexFloat] array. */
    public fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>

    /** Applies the sine function element-wise to a [ComplexDouble] array. */
    public fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    // ---- cos ----

    /** Applies the cosine function element-wise to a numeric array, returning [Double] results. */
    public fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D>

    /** Applies the cosine function element-wise to a float array, preserving single precision. */
    public fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D>

    /** Applies the cosine function element-wise to a [ComplexFloat] array. */
    public fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>

    /** Applies the cosine function element-wise to a [ComplexDouble] array. */
    public fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>
}
