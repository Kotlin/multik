package org.jetbrains.kotlinx.multik.api

import kotlin.native.concurrent.ThreadLocal

@ThreadLocal
private val _enginesProvider : HashMap<EngineType, Engine> = HashMap()

public actual val enginesProvider : Map<EngineType, Engine>
    get() = _enginesProvider

public actual fun initEnginesProvider(engines: List<Engine>) {
    engines.forEach {
        _enginesProvider[it.type] = it
    }
    Engine.loadEngine()
}