/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.Statistics
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.jni.linalg.NativeLinAlg
import org.jetbrains.kotlinx.multik.jni.math.NativeMath


public class NativeEngine : Engine() {

    override val name: String
        get() = type.name

    override val type: EngineType
        get() = NativeEngineType

    private val loader: Loader by lazy { libLoader("multik_jni") }

    init {
        if(!loader.loading) loader.load()
    }

    override fun getMath(): Math {
        return NativeMath
    }

    override fun getLinAlg(): LinAlg {
        return NativeLinAlg
    }

    override fun getStatistics(): Statistics {
        return NativeStatistics
    }
}