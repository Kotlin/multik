/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.kotlin.KEEngine
import org.jetbrains.kotlinx.multik.openblas.NativeEngine

internal actual object DefaultEngineFactory : EngineFactory {
    override fun getEngine(type: EngineType?): Engine =
        when (type) {
            null -> error("Pass NativeEngineType of KEEngineType")
            KEEngineType -> KEEngine()
            NativeEngineType -> NativeEngine()
            DefaultEngineType -> error("Default Engine doesn't link to Default Engine")
        }
}