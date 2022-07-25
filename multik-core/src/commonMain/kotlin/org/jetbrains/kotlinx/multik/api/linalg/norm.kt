/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

/**
 * Returns norm of float matrix
 */
@JvmName("normF")
public fun LinAlg.norm(mat: MultiArray<Float, D2>, norm: Norm = Norm.Fro): Float = this.linAlgEx.normF(mat, norm)

/**
 * Returns norm of double matrix
 */
@JvmName("normD")
public fun LinAlg.norm(mat: MultiArray<Double, D2>, norm: Norm = Norm.Fro): Double = this.linAlgEx.norm(mat, norm)
