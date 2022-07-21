/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Solves a linear matrix equation, or system of linear scalar equations.
 */
@JvmName("solveF")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> = this.linAlgEx.solveF(a, b)
/**
 * Solves a linear matrix equation, or system of linear scalar equations.
 */
@JvmName("solveD")
public fun <T : Number, D : Dim2> LinAlg.solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)
/**
 * Solves a linear matrix equation, or system of linear scalar equations.
 */
@JvmName("solveC")
public fun <T : Complex, D : Dim2> LinAlg.solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> = this.linAlgEx.solveC(a, b)