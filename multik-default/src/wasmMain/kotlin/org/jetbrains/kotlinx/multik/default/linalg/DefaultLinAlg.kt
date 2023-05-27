/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public actual object DefaultLinAlg : LinAlg {

    private val ktLinAlg = DefaultEngineFactory.getEngine(KEEngineType).getLinAlg()

    actual override val linAlgEx: LinAlgEx
        get() = DefaultLinAlgEx

    actual override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> = ktLinAlg.pow(mat, n)
}