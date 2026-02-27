package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Computes the PLU decomposition of a float matrix, preserving single precision.
 *
 * Decomposes [mat] into permutation **P**, lower-triangular **L**, and upper-triangular **U**
 * such that **A = P * L * U**.
 *
 * @param mat the input matrix.
 * @return a [Triple] of (P, L, U) as float [D2Array]s.
 * @see [qr] for QR decomposition.
 */
@JvmName("pluF")
public fun LinAlg.plu(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>> = this.linAlgEx.pluF(mat)

/**
 * Computes the PLU decomposition of a numeric matrix.
 *
 * Decomposes [mat] into permutation **P**, lower-triangular **L**, and upper-triangular **U**
 * such that **A = P * L * U**. The result is double-precision.
 *
 * ```
 * val a = mk.ndarray(mk[mk[2.0, 3.0], mk[5.0, 4.0]])
 * val (p, l, u) = mk.linalg.plu(a)
 * // a ≈ p dot l dot u
 * ```
 *
 * @param mat the input matrix.
 * @return a [Triple] of (P, L, U) as double-precision [D2Array]s.
 * @see [qr] for QR decomposition.
 */
@JvmName("pluD")
public fun <T : Number> LinAlg.plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> = this.linAlgEx.plu(mat)

/**
 * Computes the PLU decomposition of a complex matrix.
 *
 * Decomposes [mat] into permutation **P**, lower-triangular **L**, and upper-triangular **U**
 * such that **A = P * L * U**.
 *
 * @param mat the input matrix.
 * @return a [Triple] of (P, L, U) as complex [D2Array]s.
 * @see [qr] for QR decomposition.
 */
@JvmName("pluC")
public fun <T : Complex> LinAlg.plu(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> = this.linAlgEx.pluC(mat)
