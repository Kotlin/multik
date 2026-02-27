package org.jetbrains.kotlinx.multik.api

/**
 * Engine Provider for JS.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> = enginesStore

/**
 * Module-level mutable registry where JS engine factories store their [Engine] instances during initialization.
 *
 * Engines register themselves into this map eagerly at module load time. [enginesProvider] returns this map directly.
 */
public val enginesStore: MutableMap<EngineType, Engine> = mutableMapOf()
