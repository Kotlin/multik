/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

@JvmName("invF")
public fun LinAlg.inv(mat: MultiArray<Float, D2>): NDArray<Float, D2> = this.linAlgEx.invF(mat)

@JvmName("invD")
public fun <T : Number> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invC")
public fun <T : Complex> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.invC(mat)