package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map

/**
 * Transforms this [NDArray] of [ComplexFloat] to an [NDArray] of the real part of complex numbers.
 * Dimensions are preserved.
 *
 * @param D dimension.
 * @return [NDArray] of real portion of [ComplexFloat]
 */
@get:JvmName("reFloat")
public val <D : Dimension> MultiArray<ComplexFloat, D>.re: NDArray<Float, D>
    get() = this.map { it.re }

/**
 * Transforms this [NDArray] of [ComplexDouble] to an [NDArray] of the real part of complex numbers.
 * Dimensions are preserved.
 *
 * @param D dimension.
 * @return [NDArray] of real portion of [ComplexDouble]
 */
@get:JvmName("reDouble")
public val <D : Dimension> MultiArray<ComplexDouble, D>.re: NDArray<Double, D>
    get() = this.map { it.re }

/**
 * Transforms this [NDArray] of [ComplexFloat] to an [NDArray] of the imaginary part of complex numbers.
 * Dimensions are preserved.
 *
 * @param D dimension.
 * @return [NDArray] of imaginary portion of [ComplexFloat]
 */
@get:JvmName("imFloat")
public val <D : Dimension> MultiArray<ComplexFloat, D>.im: NDArray<Float, D>
    get() = this.map { it.im }

/**
 * Transforms this [NDArray] of [ComplexDouble] to an [NDArray] of the imaginary part of complex numbers.
 * Dimensions are preserved.
 *
 * @param D dimension.
 * @return [NDArray] of imaginary portion of [ComplexDouble]
 */
@get:JvmName("imDouble")
public val <D : Dimension> MultiArray<ComplexDouble, D>.im: NDArray<Double, D>
    get() = this.map { it.im }
