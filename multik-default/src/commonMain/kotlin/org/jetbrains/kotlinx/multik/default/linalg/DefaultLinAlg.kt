/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object DefaultLinAlg : LinAlg {

    override val linAlgEx: LinAlgEx
        get() = DefaultLinAlgEx

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        return JvmLinAlg.pow(mat, n)
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double = JvmLinAlg.norm(mat, p)
}