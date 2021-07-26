/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * Linear Algebra methods interface.
 */
public interface LinAlg {

    public val linAlgEx: LinAlgEx

    /**
     * Raise a square matrix to power [n].
     */
    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

    /**
     * Matrix ov vector norm. The default is Frobenius norm.
     */
    public fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double

    /**
     * Dot products of two arrays. Matrix product.
     */
    public fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    /**
     * Dot products of two one-dimensional arrays. Scalar product.
     */
    public fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}

public interface LinAlgEx {
    public fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2>
    public fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2>
    public fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2>


    public fun <T: Number, D: Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>
    public fun <D: Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D>
    public fun <T: Complex, D: Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>
}