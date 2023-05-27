package org.jetbrains.kotlinx.multik.api

/**
 * Engine Provider for WASM.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> = enginesStore

/**
 * Saves and initialize engine.
 */
public val enginesStore: MutableMap<EngineType, Engine> = mutableMapOf()