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
