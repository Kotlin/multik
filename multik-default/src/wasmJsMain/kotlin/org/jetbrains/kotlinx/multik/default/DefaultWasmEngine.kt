/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.DefaultEngineType
import org.jetbrains.kotlinx.multik.api.enginesStore

@ExperimentalJsExport
@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
public val initializer: EngineInitializer = EngineInitializer

public object EngineInitializer {
    init {
        enginesStore[DefaultEngineType] = DefaultEngine()
    }
}