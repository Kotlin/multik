/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.kotlin

import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.kotlin.linalg.KELinAlg
import org.jetbrains.kotlinx.multik.kotlin.math.KEMath
import org.jetbrains.kotlinx.multik.kotlin.stat.KEStatistics


public class KEEngine : Engine() {

    override val name: String
        get() = type.name

    override val type: EngineType
        get() = KEEngineType

    override fun getMath(): Math {
        return KEMath
    }

    override fun getLinAlg(): LinAlg {
        return KELinAlg
    }

    override fun getStatistics(): Statistics {
        return KEStatistics
    }
}