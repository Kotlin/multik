/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * Linear Algebra methods interface.
 */
public interface LinAlg {

    /**
     * Raise a square matrix to power [n].
     */
    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

    /**
     * Matrix ov vector norm. The default is Frobenius norm.
     */
    public fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double

    public fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<T, D2>

    public fun <T : Number, D: Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>

    /**
     * Dot products of two arrays. Matrix product.
     */
    public fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    /**
     * Dot products of two one-dimensional arrays. Scalar product.
     */
    public fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}