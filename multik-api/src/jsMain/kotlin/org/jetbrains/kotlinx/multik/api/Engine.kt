package org.jetbrains.kotlinx.multik.api


public actual val enginesProvider : Map<EngineType, Engine> = HashMap()


public actual fun initEnginesProvider(engines: List<Engine>) {
    engines.forEach {
        (enginesProvider as HashMap)[it.type] = it
    }
    Engine.loadEngine()
}