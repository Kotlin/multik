/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Extension interface for [LinAlg] for improved type support.
 */
public interface LinAlgEx {
    /**
     * Returns inverse of a double matrix from numeric matrix
     */
    public fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2>
    /**
     * Returns inverse float matrix
     */
    public fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2>
    /**
     * Returns inverse complex matrix
     */
    public fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2>

    /**
     * Solve a linear matrix equation, or system of linear scalar equations.
     */
    public fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>
    /**
     * Solve a linear matrix equation, or system of linear scalar equations.
     */
    public fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D>
    /**
     * Solve a linear matrix equation, or system of linear scalar equations.
     */
    public fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    /**
     * Returns norm of float matrix
     */
    public fun normF(mat: MultiArray<Float, D2>, norm: Norm = Norm.Fro): Float
    /**
     * Returns norm of double matrix
     */
    public fun norm(mat: MultiArray<Double, D2>, norm: Norm = Norm.Fro): Double

    /**
     * Returns QR decomposition of the numeric matrix
     */
    public fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>>
    /**
     * Returns QR decomposition of the float matrix
     */
    public fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>>
    /**
     * Returns QR decomposition of the complex matrix
     */
    public fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>>

    /**
     * Returns PLU decomposition of the numeric matrix
     */
    public fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>>
    /**
     * Returns PLU decomposition of the float matrix
     */
    public fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>>
    /**
     * Returns PLU decomposition of the complex matrix
     */
    public fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>>

    /**
     * Calculates the eigenvalues and eigenvectors of a numeric matrix
     * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
     */
    public fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>
    /**
     * Calculates the eigenvalues and eigenvectors of a float matrix
     * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
     */
    public fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>
    /**
     * Calculates the eigenvalues and eigenvectors of a complex matrix
     * @return a pair of a vector of eigenvalues and a matrix of eigenvectors
     */
    public fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>>

    /**
     * Calculates the eigenvalues of a numeric matrix.
     * @return [ComplexDouble] vector
     */
    public fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble>
    /**
     * Calculates the eigenvalues of a float matrix
     * @return [ComplexFloat] vector
     */
    public fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat>
    /**
     * Calculates the eigenvalues of a float matrix
     * @return complex vector
     */
    public fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T>

    /**
     * Dot products of two number matrices.
     */
    public fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
    /**
     * Dot products of two complex matrices.
     */
    public fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
    /**
     * Dot products of number matrix and number vector.
     */
    public fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
    /**
     * Dot products of complex matrix and complex vector.
     */
    public fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
    /**
     * Dot products of two number vectors. Scalar product.
     */
    public fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
    /**
     * Dot products of two complex vectors. Scalar product.
     */
    public fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}