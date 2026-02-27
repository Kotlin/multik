package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Computes the inverse of a float matrix.
 *
 * The input [mat] must be square and non-singular.
 *
 * @param mat a square, non-singular float matrix.
 * @return a new [NDArray] containing the inverse matrix, same shape as [mat].
 * @see [solve] for solving linear systems (preferred over explicit inversion).
 */
@JvmName("invF")
public fun LinAlg.inv(mat: MultiArray<Float, D2>): NDArray<Float, D2> = this.linAlgEx.invF(mat)

/**
 * Computes the inverse of a numeric matrix, returning a double-precision result.
 *
 * The input [mat] must be square and non-singular.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * val aInv = mk.linalg.inv(a)
 * // [[-2.0,  1.0],
 * //  [ 1.5, -0.5]]
 * ```
 *
 * @param mat a square, non-singular numeric matrix.
 * @return a new [NDArray] of [Double] containing the inverse matrix, same shape as [mat].
 * @see [solve] for solving linear systems (preferred over explicit inversion).
 */
@JvmName("invD")
public fun <T : Number> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

/**
 * Computes the inverse of a complex matrix.
 *
 * The input [mat] must be square and non-singular.
 *
 * @param mat a square, non-singular complex matrix.
 * @return a new [NDArray] containing the inverse matrix, same shape as [mat].
 * @see [solve] for solving linear systems (preferred over explicit inversion).
 */
@JvmName("invC")
public fun <T : Complex> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.invC(mat)
