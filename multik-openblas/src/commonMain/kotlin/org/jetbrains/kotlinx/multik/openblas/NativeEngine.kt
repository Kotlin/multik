/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas

import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.openblas.linalg.NativeLinAlg
import org.jetbrains.kotlinx.multik.openblas.math.NativeMath
import org.jetbrains.kotlinx.multik.openblas.stat.NativeStatistics


public open class NativeEngine : Engine() {

    override val name: String
        get() = type.name

    override val type: EngineType
        get() = NativeEngineType

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