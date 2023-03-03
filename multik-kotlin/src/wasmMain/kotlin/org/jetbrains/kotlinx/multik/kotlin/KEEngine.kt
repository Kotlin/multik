package org.jetbrains.kotlinx.multik.kotlin

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.enginesStore


@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
public val initializer: EngineInitializer = EngineInitializer

public object EngineInitializer {
    init {
        enginesStore[KEEngineType] = KEEngine()
    }
}