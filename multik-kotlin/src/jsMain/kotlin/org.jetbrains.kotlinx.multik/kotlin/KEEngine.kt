package org.jetbrains.kotlinx.multik.kotlin

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.enginesStore

@ExperimentalJsExport
@ExperimentalStdlibApi
@Suppress("unused", "DEPRECATION")
@EagerInitialization
@JsExport
public val initializer: dynamic = EngineInitializer

public object EngineInitializer {
    init {
        enginesStore[KEEngineType] = KEEngine()
    }
}
