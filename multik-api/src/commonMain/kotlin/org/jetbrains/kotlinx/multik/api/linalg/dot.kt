/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Dot products of two number matrices.
 */
@JvmName("dotMMNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMM(a, b)

/**
 * Dot products of two complex matrices.
 */
@JvmName("dotMMComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.dotMMComplex(a, b)

/**
 * Dot products of number matrix and number vector.
 */
@JvmName("dotMVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMV(a, b)

/**
 * Dot products of complex matrix and complex vector.
 */
@JvmName("dotMVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> = this.linAlgEx.dotMVComplex(a, b)

/**
 * Dot products of two number vectors. Scalar product.
 */
@JvmName("dotVVNumber")
public fun <T : Number> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVV(a, b)

/**
 * Dot products of two complex vectors. Scalar product.
 */
@JvmName("dotVVComplex")
public fun <T : Complex> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = this.linAlgEx.dotVVComplex(a, b)
