package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.engines

@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
private val InitHook = EngineInitializer

private object EngineInitializer {
    init {
        engines[NativeEngineType] = NativeEngine()
    }
}