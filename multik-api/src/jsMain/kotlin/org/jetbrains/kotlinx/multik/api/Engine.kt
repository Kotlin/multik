package org.jetbrains.kotlinx.multik.api

public actual fun enginesProvider(): Map<EngineType, Engine> = enginesStore

public val enginesStore: MutableMap<EngineType, Engine> = mutableMapOf<EngineType, Engine>()
