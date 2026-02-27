package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Computes the matrix product of two numeric matrices.
 *
 * The number of columns in [a] must equal the number of rows in [b].
 * For an (m × n) matrix multiplied by an (n × p) matrix, the result is an (m × p) matrix.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
 * val b = mk.ndarray(mk[mk[5, 6], mk[7, 8]])
 * val c = mk.linalg.dot(a, b)
 * // [[19, 22],
 * //  [43, 50]]
 * ```
 *
 * @param a the left-hand matrix.
 * @param b the right-hand matrix.
 * @return a new [NDArray] of shape (m × p).
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotMMNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMM(a, b)

/**
 * Computes the matrix product of two complex matrices.
 *
 * The number of columns in [a] must equal the number of rows in [b].
 * For an (m × n) matrix multiplied by an (n × p) matrix, the result is an (m × p) matrix.
 *
 * @param a the left-hand matrix.
 * @param b the right-hand matrix.
 * @return a new [NDArray] of shape (m × p).
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotMMComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMMComplex(a, b)

/**
 * Computes the product of a numeric matrix and a numeric vector.
 *
 * The vector length must equal the number of columns in [a].
 * For an (m × n) matrix multiplied by a vector of length n, the result is a vector of length m.
 *
 * ```
 * val mat = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * val vec = mk.ndarray(mk[10.0, 20.0])
 * val result = mk.linalg.dot(mat, vec) // [50.0, 110.0]
 * ```
 *
 * @param a the left-hand matrix.
 * @param b the right-hand vector.
 * @return a new [NDArray] vector of length m.
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotMVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMV(a, b)

/**
 * Computes the product of a complex matrix and a complex vector.
 *
 * The vector length must equal the number of columns in [a].
 * For an (m × n) matrix multiplied by a vector of length n, the result is a vector of length m.
 *
 * @param a the left-hand matrix.
 * @param b the right-hand vector.
 * @return a new [NDArray] vector of length m.
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotMVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMVComplex(a, b)

/**
 * Computes the inner (scalar) product of two numeric vectors.
 *
 * Both vectors must have the same length. The result is a scalar value equal to the sum
 * of element-wise products.
 *
 * ```
 * val u = mk.ndarray(mk[1.0, 2.0, 3.0])
 * val v = mk.ndarray(mk[4.0, 5.0, 6.0])
 * val scalar = mk.linalg.dot(u, v) // 32.0  (1*4 + 2*5 + 3*6)
 * ```
 *
 * @param a the left-hand vector.
 * @param b the right-hand vector, must have the same length as [a].
 * @return the scalar product of the two vectors.
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotVVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVV(a, b)

/**
 * Computes the inner (scalar) product of two complex vectors.
 *
 * Both vectors must have the same length. The result is a scalar value equal to the sum
 * of element-wise products.
 *
 * @param a the left-hand vector.
 * @param b the right-hand vector, must have the same length as [a].
 * @return the scalar product of the two vectors.
 * @see [MultiArray.dot] for the infix equivalent: `a dot b`.
 */
@JvmName("dotVVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVVComplex(a, b)
