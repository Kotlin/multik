package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import kotlin.jvm.JvmName

/**
 * Computes a norm for a float vector.
 *
 * The vector is internally stacked into a matrix for computation.
 * Only accepts `Float` arrays — convert integer arrays with `toType<Float>()` first.
 *
 * @param mat the input vector.
 * @param norm the [Norm] type to compute (default: [Norm.Fro]).
 * @return the scalar norm value.
 */
@JvmName("normFV")
public fun LinAlg.norm(mat: MultiArray<Float, D1>, norm: Norm = Norm.Fro): Float =
    this.linAlgEx.normF(mk.stack(mat, mk.zeros(mat.size)), norm)

/**
 * Computes a norm for a float matrix.
 *
 * Only accepts `Float` arrays — convert integer arrays with `toType<Float>()` first.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1f, 2f], mk[3f, 4f]])
 * mk.linalg.norm(a)           // Frobenius norm
 * mk.linalg.norm(a, Norm.N1)  // 1-norm (max column sum)
 * ```
 *
 * @param mat the input matrix.
 * @param norm the [Norm] type to compute (default: [Norm.Fro]).
 * @return the scalar norm value.
 */
@JvmName("normF")
public fun LinAlg.norm(mat: MultiArray<Float, D2>, norm: Norm = Norm.Fro): Float = this.linAlgEx.normF(mat, norm)

/**
 * Computes a norm for a double vector.
 *
 * The vector is internally stacked into a matrix for computation.
 * Only accepts `Double` arrays — convert integer arrays with `toType<Double>()` first.
 *
 * @param mat the input vector.
 * @param norm the [Norm] type to compute (default: [Norm.Fro]).
 * @return the scalar norm value.
 */
@JvmName("normDV")
public fun LinAlg.norm(mat: MultiArray<Double, D1>, norm: Norm = Norm.Fro): Double =
    this.linAlgEx.norm(mk.stack(mat, mk.zeros(mat.size)), norm)

/**
 * Computes a norm for a double matrix.
 *
 * Only accepts `Double` arrays — convert integer arrays with `toType<Double>()` first.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * mk.linalg.norm(a)                // Frobenius: sqrt(30) ≈ 5.477
 * mk.linalg.norm(a, Norm.N1)       // max column sum: 6.0
 * mk.linalg.norm(a, Norm.Inf)      // max row sum: 7.0
 * mk.linalg.norm(a, Norm.Max)      // max |element|: 4.0
 * ```
 *
 * @param mat the input matrix.
 * @param norm the [Norm] type to compute (default: [Norm.Fro]).
 * @return the scalar norm value.
 * @see [Norm] for available norm types.
 */
@JvmName("normD")
public fun LinAlg.norm(mat: MultiArray<Double, D2>, norm: Norm = Norm.Fro): Double = this.linAlgEx.norm(mat, norm)
