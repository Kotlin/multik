package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.JvmEngineType
import org.jetbrains.kotlinx.multik.api.engines


@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
private val InitHook = EngineInitializer

private object EngineInitializer {
    init {
        engines[JvmEngineType] = JvmEngine()
    }
}