/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

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
 * Calculates the eigenvalues and eigenvectors of a float matrix
 * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
 */
@JvmName("eigF")
public fun LinAlg.eig(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
    this.linAlgEx.eigF(mat)

/**
 * Calculates the eigenvalues and eigenvectors of a numeric matrix
 * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
 */
@JvmName("eig")
public fun <T : Number> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
    this.linAlgEx.eig(mat)

/**
 * Calculates the eigenvalues and eigenvectors of a complex matrix
 * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
 */
@JvmName("eigC")
public fun <T : Complex> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> =
    this.linAlgEx.eigC(mat)

/**
 * Calculates the eigenvalues of a float matrix
 * @return [ComplexFloat] vector
 */
@JvmName("eigValsF")
public fun LinAlg.eigVals(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> = this.linAlgEx.eigValsF(mat)

/**
 * Calculates the eigenvalues of a numeric matrix.
 * @return [ComplexDouble] vector
 */
@JvmName("eigVals")
public fun <T : Number> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> = this.linAlgEx.eigVals(mat)

/**
 * Calculates the eigenvalues of a float matrix
 * @return complex vector
 */
@JvmName("eigValsC")
public fun <T : Complex> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<T> = this.linAlgEx.eigValsC(mat)