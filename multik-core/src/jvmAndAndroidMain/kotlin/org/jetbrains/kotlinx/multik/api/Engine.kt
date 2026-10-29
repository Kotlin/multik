package org.jetbrains.kotlinx.multik.api

import java.util.*
import java.util.concurrent.ConcurrentHashMap

/**
 * Engine Provider for JVM.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> {
    val engineList = ServiceLoader.load(Engine::class.java).toList()

    if (engineList.isEmpty()) {
        error(
            """Fail to find engine. Consider to add one of the following dependencies: 
 - multik-default
 - multik-kotlin
 - multik-openblas"""
        )
    }

    return ConcurrentHashMap(engineList.associateBy { it.type })
}
