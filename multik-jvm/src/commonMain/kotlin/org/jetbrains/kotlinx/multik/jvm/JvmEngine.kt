/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg


public class JvmEngine : Engine() {

    override val name: String
        get() = type.name

    override val type: EngineType
        get() = JvmEngineType

    override fun getMath(): Math {
        return JvmMath
    }

    override fun getLinAlg(): LinAlg {
        return JvmLinAlg
    }

    override fun getStatistics(): Statistics {
        return JvmStatistics
    }
}