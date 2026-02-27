package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Computes the SVD decomposition of a float matrix such that **A = U * S * V^T**.
 *
 * @param mat the input matrix.
 * @return a [Triple] of (U, S, V^T) where S is a [D1Array] of singular values.
 * @see [qr] for QR decomposition.
 */
@ExperimentalMultikApi
@JvmName("svdF")
public fun LinAlg.svd(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>> = this.linAlgEx.svdF(mat)

/**
 * Computes the SVD decomposition of a numeric matrix such that **A = U * S * V^T**.
 *
 * The result is double-precision.
 *
 * ```
 * @OptIn(ExperimentalMultikApi::class)
 * val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
 * val (u, s, vt) = mk.linalg.svd(a)
 * // u: left singular vectors
 * // s: singular values [s1, s2]
 * // vt: right singular vectors (transposed)
 * ```
 *
 * @param mat the input matrix.
 * @return a [Triple] of (U, S, V^T) as double-precision arrays where S is a [D1Array] of singular values.
 * @see [qr] for QR decomposition.
 */
@ExperimentalMultikApi
@JvmName("svdD")
public fun <T : Number> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>> = this.linAlgEx.svd(mat)

/**
 * Computes the SVD decomposition of a complex matrix such that **A = U * S * V^H**.
 *
 * @param mat the input matrix.
 * @return a [Triple] of (U, S, V^H) where S is a [D1Array] of singular values.
 * @see [qr] for QR decomposition.
 */
@ExperimentalMultikApi
@JvmName("svdC")
public fun <T : Complex> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>> = this.linAlgEx.svdC(mat)
