/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.openblas.linalg.NativeLinAlg
import org.jetbrains.kotlinx.multik.openblas.math.NativeMath
import org.jetbrains.kotlinx.multik.openblas.stat.NativeStatistics

public class JvmNativeEngine: NativeEngine() {
    private val loader: Loader by lazy { libLoader("multik_jni") }

    override fun getMath(): Math {
        if(!loader.loading) loader.load()
        return NativeMath
    }

    override fun getLinAlg(): LinAlg {
        if(!loader.loading) loader.load()
        return NativeLinAlg
    }

    override fun getStatistics(): Statistics {
        if(!loader.loading) loader.load()
        return NativeStatistics
    }
}