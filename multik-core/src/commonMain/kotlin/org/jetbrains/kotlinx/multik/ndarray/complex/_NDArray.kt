package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import kotlin.jvm.JvmName

/**
 * Extracts the real parts of all elements, returning a new [NDArray] of [Float] with the same shape.
 *
 * ```
 * val a = mk.ndarray(mk[ComplexFloat(1f, 2f), ComplexFloat(3f, 4f)])
 * a.re // [1.0, 3.0]
 * ```
 *
 * @see im
 */
@get:JvmName("reFloat")
public val <D : Dimension> MultiArray<ComplexFloat, D>.re: NDArray<Float, D>
    get() = this.map { it.re }

/**
 * Extracts the real parts of all elements, returning a new [NDArray] of [Double] with the same shape.
 *
 * @see im
 */
@get:JvmName("reDouble")
public val <D : Dimension> MultiArray<ComplexDouble, D>.re: NDArray<Double, D>
    get() = this.map { it.re }

/**
 * Extracts the imaginary parts of all elements, returning a new [NDArray] of [Float] with the same shape.
 *
 * ```
 * val a = mk.ndarray(mk[ComplexFloat(1f, 2f), ComplexFloat(3f, 4f)])
 * a.im // [2.0, 4.0]
 * ```
 *
 * @see re
 */
@get:JvmName("imFloat")
public val <D : Dimension> MultiArray<ComplexFloat, D>.im: NDArray<Float, D>
    get() = this.map { it.im }

/**
 * Extracts the imaginary parts of all elements, returning a new [NDArray] of [Double] with the same shape.
 *
 * @see re
 */
@get:JvmName("imDouble")
public val <D : Dimension> MultiArray<ComplexDouble, D>.im: NDArray<Double, D>
    get() = this.map { it.im }

/**
 * Returns a new array where each element is the complex conjugate of the original.
 * The shape is preserved.
 *
 * @see ComplexDouble.conjugate
 */
@JvmName("conjDouble")
public fun <D: Dimension> MultiArray<ComplexDouble, D>.conj(): MultiArray<ComplexDouble, D> = this.map { it.conjugate() }

/**
 * Returns a new array where each element is the complex conjugate of the original.
 * The shape is preserved.
 *
 * @see ComplexFloat.conjugate
 */
@JvmName("conjFloat")
public fun <D: Dimension> MultiArray<ComplexFloat, D>.conj(): MultiArray<ComplexFloat, D> = this.map { it.conjugate() }
