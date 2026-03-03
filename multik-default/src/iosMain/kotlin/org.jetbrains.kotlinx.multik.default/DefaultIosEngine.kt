package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.DefaultEngineType
import org.jetbrains.kotlinx.multik.api.engines

@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
private val InitHook = EngineInitializer

private object EngineInitializer {
    init {
        engines.value[DefaultEngineType] = DefaultEngine()
    }
}
