package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Computes the matrix product of two numeric matrices.
 *
 * The number of columns in this matrix must equal the number of rows in [b].
 * For an (m × n) matrix multiplied by an (n × p) matrix, the result is an (m × p) matrix.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
 * val b = mk.ndarray(mk[mk[5, 6], mk[7, 8]])
 * val c = a dot b
 * // [[19, 22],
 * //  [43, 50]]
 * ```
 *
 * @param b the right-hand matrix.
 * @return a new [NDArray] of shape (m × p).
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefMMNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMM(this, b)

/**
 * Computes the matrix product of two complex matrices.
 *
 * The number of columns in this matrix must equal the number of rows in [b].
 * For an (m × n) matrix multiplied by an (n × p) matrix, the result is an (m × p) matrix.
 *
 * @param b the right-hand matrix.
 * @return a new [NDArray] of shape (m × p).
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefMMComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMMComplex(this, b)

/**
 * Computes the product of a numeric matrix and a numeric vector.
 *
 * The vector length must equal the number of columns in this matrix.
 * For an (m × n) matrix multiplied by a vector of length n, the result is a vector of length m.
 *
 * ```
 * val mat = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * val vec = mk.ndarray(mk[10.0, 20.0])
 * val result = mat dot vec // [50.0, 110.0]
 * ```
 *
 * @param b the right-hand vector.
 * @return a new [NDArray] vector of length m.
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefMVNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMV(this, b)

/**
 * Computes the product of a complex matrix and a complex vector.
 *
 * The vector length must equal the number of columns in this matrix.
 * For an (m × n) matrix multiplied by a vector of length n, the result is a vector of length m.
 *
 * @param b the right-hand vector.
 * @return a new [NDArray] vector of length m.
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefMVComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMVComplex(this, b)

/**
 * Computes the inner (scalar) product of two numeric vectors.
 *
 * Both vectors must have the same length. The result is a scalar value equal to the sum
 * of element-wise products.
 *
 * ```
 * val u = mk.ndarray(mk[1.0, 2.0, 3.0])
 * val v = mk.ndarray(mk[4.0, 5.0, 6.0])
 * val scalar = u dot v // 32.0  (1*4 + 2*5 + 3*6)
 * ```
 *
 * @param b the right-hand vector, must have the same length as this vector.
 * @return the scalar product of the two vectors.
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefVVNumber")
public infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVV(this, b)

/**
 * Computes the inner (scalar) product of two complex vectors.
 *
 * Both vectors must have the same length. The result is a scalar value equal to the sum
 * of element-wise products.
 *
 * @param b the right-hand vector, must have the same length as this vector.
 * @return the scalar product of the two vectors.
 * @see [LinAlg.dot] for the functional-style equivalent via `mk.linalg.dot(a, b)`.
 */
@JvmName("dotDefVVComplex")
public infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVVComplex(this, b)
