package org.jetbrains.kotlinx.multik.kotlin

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.engines


@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
private val InitHook = EngineInitializer

private object EngineInitializer {
    init {
        engines.value[KEEngineType] = KEEngine()
    }
}