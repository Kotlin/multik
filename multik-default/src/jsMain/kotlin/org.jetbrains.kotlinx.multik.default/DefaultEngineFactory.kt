/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.kotlin.KEEngine

internal actual object DefaultEngineFactory : EngineFactory {
    override fun getEngine(type: EngineType?): Engine =
        when (type) {
            null, KEEngineType, DefaultEngineType -> KEEngine()
            NativeEngineType -> error("Don't exist native engine for iOS targets")
        }
}