/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.default.linalg.DefaultLinAlg
import org.jetbrains.kotlinx.multik.default.math.DefaultMath

public class DefaultEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = DefaultEngineType

    override fun getMath(): Math {
        return DefaultMath
    }

    override fun getLinAlg(): LinAlg {
        return DefaultLinAlg
    }

    override fun getStatistics(): Statistics {
        return DefaultStatistics
    }
}