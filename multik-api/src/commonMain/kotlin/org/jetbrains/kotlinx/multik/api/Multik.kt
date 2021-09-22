/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import kotlin.native.concurrent.ThreadLocal
import org.jetbrains.kotlinx.multik.api.Multik.engine
import org.jetbrains.kotlinx.multik.api.Multik.engines
import org.jetbrains.kotlinx.multik.api.Multik.linalg
import org.jetbrains.kotlinx.multik.api.Multik.math
import org.jetbrains.kotlinx.multik.api.Multik.stat
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg

/**
 * Abbreviated name for [Multik].
 */
public typealias mk = Multik


/**
 * The basic object through which calls all ndarray functions. Gives access to ndarray creation and interfaces [Math],
 * [LinAlg] and [Statistics].
 * Calling [Multik] will load the engine. The default is "DEFAULT".
 * If no engine is found, then an exception is thrown only when you call an implementation that requires the engine.
 *
 * Note: Through [Multik], you can set your own interface implementation.
 *
 * @property engine currently used engine.
 * @property engines list of engines.
 * @property math returns the [Math] implementation of the corresponding engine.
 * @property linalg returns the [LinAlg] implementation of the corresponding engine.
 * @property stat returns the [Statistics] implementation of the corresponding engine.
 */
@ThreadLocal
public object Multik {
    public val engine: String? get() = Engine.getDefaultEngine()

    private val _engines: MutableMap<String, EngineType> = mutableMapOf(
        "DEFAULT" to DefaultEngineType,
        "JVM" to JvmEngineType,
        "NATIVE" to NativeEngineType
    )

    public val engines: Map<String, EngineType>
        get() = _engines

    public val math: Math get() = Engine.getMath()
    public val linalg: LinAlg get() = Engine.getLinAlg()
    public val stat: Statistics get() = Engine.getStatistics()

    /**
     * Adds engine to [engines].
     */
    public fun addEngine(type: EngineType) {
        if (!_engines.containsKey(type.name)) {
            _engines[type.name] = type
        }
    }

    /**
     * Sets the engine of type [type] as the current implementation.
     */
    public fun setEngine(type: EngineType) {
        if (type.name in engines)
            Engine.setDefaultEngine(type)
    }

    /**
     * Returns a list of [elements]. Sugar for easy array creation.
     */
    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
