package org.jetbrains.kotlinx.multik.api

import kotlin.reflect.full.createInstance


public actual val enginesProvider : Set<Engine> get() {
        val engineClassNames = listOf(
            "org.jetbrains.kotlinx.multik.jvm.JvmEngine",
            "org.jetbrains.kotlinx.multik.jni.NativeEngine",
            "org.jetbrains.kotlinx.multik.default.DefaultEngine")
        val engines = mutableSetOf<Engine>()
        engineClassNames.forEach { e ->
            try {
                val instance = Class.forName(e).kotlin.createInstance() as Engine
                engines.add(instance)
            } catch (t: Throwable){ }
        }
        return engines.toSet()
    }

public actual fun initEnginesProvider(engines: List<Engine>) {
}