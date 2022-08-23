/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Returns SVD decomposition of the float matrix
 */
@ExperimentalMultikApi
@JvmName("svdF")
public fun LinAlg.svd(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>> = this.linAlgEx.svdF(mat)

/**
 * Returns SVD decomposition of the numeric matrix
 */
@ExperimentalMultikApi
@JvmName("svdD")
public fun <T : Number> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>> = this.linAlgEx.svd(mat)

/**
 * Returns SVD decomposition of the complex matrix
 */
@ExperimentalMultikApi
@JvmName("svdC")
public fun <T : Complex> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>> = this.linAlgEx.svdC(mat)