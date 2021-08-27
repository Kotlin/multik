package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Dot products of two arrays. Matrix product.
 */
@JvmName("dotMMNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMM(a, b)

@JvmName("dotMMComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMMComplex(a, b)

/**
 * Dot products of two arrays. Matrix product.
 */
@JvmName("dotMVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMV(a, b)

@JvmName("dotMVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMVComplex(a, b)

/**
 * Dot products of two one-dimensional arrays. Scalar product.
 */
@JvmName("dotVVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVV(a, b)

@JvmName("dotVVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVVComplex(a, b)
