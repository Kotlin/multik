package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Computes the eigenvalues and eigenvectors of a float matrix.
 *
 * The input [mat] must be square. Results are complex because eigenvalues of a real matrix
 * can be complex. Not supported by the pure Kotlin engine — use the OpenBLAS or default engine.
 *
 * @param mat a square float matrix to decompose.
 * @return a [Pair] of (eigenvalues, eigenvectors) where eigenvalues is a [D1Array] and
 * eigenvectors is a [D2Array] whose columns are the corresponding eigenvectors.
 * @see [eigVals] to compute eigenvalues only (more efficient when vectors are not needed).
 */
@JvmName("eigF")
public fun LinAlg.eig(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
    this.linAlgEx.eigF(mat)

/**
 * Computes the eigenvalues and eigenvectors of a numeric matrix.
 *
 * The input [mat] must be square. Results are complex because eigenvalues of a real matrix
 * can be complex. Not supported by the pure Kotlin engine — use the OpenBLAS or default engine.
 *
 * ```
 * val a = mk.ndarray(mk[mk[4.0, -2.0], mk[1.0, 1.0]])
 * val (values, vectors) = mk.linalg.eig(a)
 * // values: [3.0+0.0i, 2.0+0.0i]
 * // vectors: columns are eigenvectors for each eigenvalue
 * ```
 *
 * @param mat a square numeric matrix to decompose.
 * @return a [Pair] of (eigenvalues, eigenvectors) where eigenvalues is a [D1Array] of [ComplexDouble] and
 * eigenvectors is a [D2Array] of [ComplexDouble] whose columns are the corresponding eigenvectors.
 * @see [eigVals] to compute eigenvalues only (more efficient when vectors are not needed).
 */
@JvmName("eig")
public fun <T : Number> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
    this.linAlgEx.eig(mat)

/**
 * Computes the eigenvalues and eigenvectors of a complex matrix.
 *
 * The input [mat] must be square. Not supported by the pure Kotlin engine — use the OpenBLAS
 * or default engine.
 *
 * @param mat a square complex matrix to decompose.
 * @return a [Pair] of (eigenvalues, eigenvectors) where eigenvalues is a [D1Array] and
 * eigenvectors is a [D2Array] whose columns are the corresponding eigenvectors.
 * @see [eigVals] to compute eigenvalues only (more efficient when vectors are not needed).
 */
@JvmName("eigC")
public fun <T : Complex> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> =
    this.linAlgEx.eigC(mat)

/**
 * Computes only the eigenvalues of a float matrix (without eigenvectors).
 *
 * The input [mat] must be square. More efficient than [eig] when eigenvectors are not needed.
 * Not supported by the pure Kotlin engine — use the OpenBLAS or default engine.
 *
 * @param mat a square float matrix.
 * @return a [D1Array] of [ComplexFloat] eigenvalues.
 * @see [eig] to compute both eigenvalues and eigenvectors.
 */
@JvmName("eigValsF")
public fun LinAlg.eigVals(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> = this.linAlgEx.eigValsF(mat)

/**
 * Computes only the eigenvalues of a numeric matrix (without eigenvectors).
 *
 * The input [mat] must be square. Results are complex because eigenvalues of a real matrix
 * can be complex. More efficient than [eig] when eigenvectors are not needed.
 * Not supported by the pure Kotlin engine — use the OpenBLAS or default engine.
 *
 * ```
 * val a = mk.ndarray(mk[mk[2.0, 1.0], mk[1.0, 2.0]])
 * val eigenvalues = mk.linalg.eigVals(a) // [3.0+0.0i, 1.0+0.0i]
 * ```
 *
 * @param mat a square numeric matrix.
 * @return a [D1Array] of [ComplexDouble] eigenvalues.
 * @see [eig] to compute both eigenvalues and eigenvectors.
 */
@JvmName("eigVals")
public fun <T : Number> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> = this.linAlgEx.eigVals(mat)

/**
 * Computes only the eigenvalues of a complex matrix (without eigenvectors).
 *
 * The input [mat] must be square. More efficient than [eig] when eigenvectors are not needed.
 * Not supported by the pure Kotlin engine — use the OpenBLAS or default engine.
 *
 * @param mat a square complex matrix.
 * @return a [D1Array] of complex eigenvalues.
 * @see [eig] to compute both eigenvalues and eigenvectors.
 */
@JvmName("eigValsC")
public fun <T : Complex> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<T> = this.linAlgEx.eigValsC(mat)
