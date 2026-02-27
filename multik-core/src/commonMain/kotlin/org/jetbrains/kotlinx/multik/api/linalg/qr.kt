package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Computes the QR decomposition of a float matrix, preserving single precision.
 *
 * Decomposes [mat] into orthogonal **Q** and upper-triangular **R** such that **A = QR**.
 *
 * @param mat the input matrix.
 * @return a [Pair] of (Q, R) as float [D2Array]s.
 * @see [plu] for PLU decomposition.
 */
@JvmName("qrF")
public fun LinAlg.qr(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> = this.linAlgEx.qrF(mat)

/**
 * Computes the QR decomposition of a numeric matrix into orthogonal **Q** and upper-triangular **R**.
 *
 * Decomposes [mat] such that **A = QR**. The result is double-precision.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * val (q, r) = mk.linalg.qr(a)
 * // q: orthogonal matrix (Q^T * Q ≈ I)
 * // r: upper-triangular matrix
 * // a ≈ q dot r
 * ```
 *
 * @param mat the input matrix.
 * @return a [Pair] of (Q, R) as double-precision [D2Array]s.
 * @see [plu] for PLU decomposition.
 */
@JvmName("qrD")
public fun <T : Number> LinAlg.qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> = this.linAlgEx.qr(mat)

/**
 * Computes the QR decomposition of a complex matrix.
 *
 * Decomposes [mat] into unitary **Q** and upper-triangular **R** such that **A = QR**.
 *
 * @param mat the input matrix.
 * @return a [Pair] of (Q, R) as complex [D2Array]s.
 * @see [plu] for PLU decomposition.
 */
@JvmName("qrC")
public fun <T : Complex> LinAlg.qr(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> = this.linAlgEx.qrC(mat)
