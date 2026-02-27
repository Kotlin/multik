package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.api.Multik.engine
import org.jetbrains.kotlinx.multik.api.Multik.engines
import org.jetbrains.kotlinx.multik.api.Multik.linalg
import org.jetbrains.kotlinx.multik.api.Multik.math
import org.jetbrains.kotlinx.multik.api.Multik.stat
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics

/**
 * Abbreviated name for [Multik]. This is the primary entry point for all Multik operations.
 */
public typealias mk = Multik


/**
 * Main entry point for the Multik library providing ndarray creation and access to
 * [Math], [LinAlg], and [Statistics] operations.
 *
 * The engine is loaded lazily on first use. The default engine is `DEFAULT` (which picks
 * `NativeEngine` on supported platforms, falling back to `KEEngine`). If no engine is available,
 * an [EngineMultikException] is thrown when an engine-dependent operation is called.
 *
 * ```
 * val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
 * val b = mk.zeros<Double>(3, 3)
 * val sum = mk.math.sum(a)
 * ```
 *
 * @property engine the name of the currently active engine, or `null` if none is loaded.
 * @property engines the map of registered engine names to their [EngineType]s.
 * @property math the [Math] implementation from the current engine.
 * @property linalg the [LinAlg] implementation from the current engine.
 * @property stat the [Statistics] implementation from the current engine.
 */
public object Multik {
    public val engine: String? get() = Engine.getDefaultEngine()

    private val _engines: MutableMap<String, EngineType> = mutableMapOf(
        "DEFAULT" to DefaultEngineType,
        "KOTLIN" to KEEngineType,
        "NATIVE" to NativeEngineType
    )

    public val engines: Map<String, EngineType>
        get() = _engines

    public val math: Math get() = Engine.getMath()
    public val linalg: LinAlg get() = Engine.getLinAlg()
    public val stat: Statistics get() = Engine.getStatistics()

    /**
     * Registers a custom engine type so it can be selected via [setEngine].
     *
     * @param type the [EngineType] to register.
     */
    public fun addEngine(type: EngineType) {
        if (!_engines.containsKey(type.name)) {
            _engines[type.name] = type
        }
    }

    /**
     * Switches the active engine to the specified [type].
     *
     * @param type the [EngineType] to activate (must be registered in [engines]).
     */
    public fun setEngine(type: EngineType) {
        if (type.name in engines)
            Engine.setDefaultEngine(type)
    }

    /**
     * Creates a list from the given [elements] for convenient nested array construction.
     *
     * ```
     * mk.ndarray(mk[mk[1, 2], mk[3, 4]])
     * ```
     */
    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
