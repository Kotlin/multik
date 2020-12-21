/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.api.*


public class NativeEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return NativeEngine
    }
}

public object NativeEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = NativeEngineType

    private val loader: Loader by lazy { Loader("multik_jni") }

    init {
        loader.load()
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