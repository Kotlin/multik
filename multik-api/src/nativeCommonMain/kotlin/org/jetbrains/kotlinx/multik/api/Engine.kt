package org.jetbrains.kotlinx.multik.api

public actual fun enginesProvider(): Map<EngineType, Engine> = engines

public val engines: MutableMap<EngineType, Engine> by lazy {
    mutableMapOf<EngineType, Engine>()
}
