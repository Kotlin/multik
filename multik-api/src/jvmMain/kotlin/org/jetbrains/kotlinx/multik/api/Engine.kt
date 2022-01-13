package org.jetbrains.kotlinx.multik.api

import kotlin.reflect.full.createInstance


public actual val enginesProvider : Map<EngineType, Engine> get() {
        val engineClassNames = listOf(
            "org.jetbrains.kotlinx.multik.jvm.JvmEngine",
            "org.jetbrains.kotlinx.multik.jni.NativeEngine",
            "org.jetbrains.kotlinx.multik.default.DefaultEngine")
        val engines = mutableMapOf<EngineType, Engine>()
        engineClassNames.forEach { e ->
            try {
                val instance = Class.forName(e).kotlin.createInstance() as Engine
                engines[instance.type] = instance
            } catch (t: Throwable){ }
        }
        return engines
    }

public actual fun initEnginesProvider(engines: List<Engine>) {
}