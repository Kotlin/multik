package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.JvmEngineType
import org.jetbrains.kotlinx.multik.api.enginesStore

@Suppress("unused")
public val initializer: EngineInitializer = EngineInitializer

public object EngineInitializer {
    init {
        enginesStore[JvmEngineType] = JvmEngine()
    }
}