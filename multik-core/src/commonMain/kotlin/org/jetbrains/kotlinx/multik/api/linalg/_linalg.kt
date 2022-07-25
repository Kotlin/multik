/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Returns the matrix product of two numeric matrices.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefMMNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMM(this, b)

/**
 * Returns the matrix product of two complex matrices.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefMMComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMMComplex(this, b)

/**
 * Returns the matrix product of a numeric matrix and a numeric vector.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefMVNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMV(this, b)

/**
 * Returns the matrix product of a complex matrix and a complex vector.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefMVComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMVComplex(this, b)

/**
 * Returns the product of two numeric vectors.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefVVNumber")
public infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVV(this, b)

/**
 * Returns the product of two complex vectors.
 *
 * same as [LinAlg.dot]
 */
@JvmName("dotDefVVComplex")
public infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVVComplex(this, b)