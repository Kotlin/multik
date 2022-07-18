/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray


/**
 * Linear Algebra methods interface.
 */
public interface LinAlg {

    public val linAlgEx: LinAlgEx

    /**
     * Raise a square matrix to power [n].
     */
    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

//    /**
//     * Matrix ov vector norm. The default is Frobenius norm.
//     */
//    public fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double
}