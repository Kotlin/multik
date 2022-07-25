/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray


/**
 * Linear Algebra methods interface.
 */
public interface LinAlg {

    /**
     * instance of [LinAlgEx]
     */
    public val linAlgEx: LinAlgEx

    /**
     * Raise a square matrix to power [n].
     */
    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

}