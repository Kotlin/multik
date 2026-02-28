package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import kotlin.concurrent.Volatile

/**
 * Sealed hierarchy of engine types identifying available computational backends.
 *
 * @param name the engine type name (e.g., "DEFAULT", "KOTLIN", "NATIVE").
 */
public sealed class EngineType(public val name: String)

/** Engine type for the default implementation that picks the best available backend. */
public object DefaultEngineType : EngineType("DEFAULT")

/** Engine type for the pure Kotlin implementation (all platforms). */
public object KEEngineType : EngineType("KOTLIN")

/** Engine type for the OpenBLAS-based native implementation (JVM + desktop Native only). */
public object NativeEngineType : EngineType("NATIVE")

/**
 * Platform-specific function that discovers and returns available [Engine] implementations.
 *
 * On JVM uses `ServiceLoader`, on Native uses `@EagerInitialization`, on JS/WASM uses module-level registration.
 */
public expect fun enginesProvider(): Map<EngineType, Engine>

/**
 * Abstract base class for computational backends providing [Math], [LinAlg], and [Statistics] implementations.
 *
 * Engines are loaded automatically when [Multik] is first used. The default engine is `DEFAULT`
 * (which delegates to `NativeEngine` when available, falling back to `KEEngine`).
 * Use [Multik.setEngine] to switch engines at runtime.
 *
 * @property name the engine name.
 * @property type the [EngineType] identifying this engine.
 * @see [Multik] for the main entry point.
 */
public abstract class Engine {

    protected abstract val name: String

    public abstract val type: EngineType

    /** Returns the [Math] implementation provided by this engine. */
    public abstract fun getMath(): Math

    /** Returns the [LinAlg] implementation provided by this engine. */
    public abstract fun getLinAlg(): LinAlg

    /** Returns the [Statistics] implementation provided by this engine. */
    public abstract fun getStatistics(): Statistics

    internal companion object : Engine() {

        private val enginesProvider: Map<EngineType, Engine> = enginesProvider()

        @Volatile
        private var defaultEngine: EngineType? = null

        fun loadEngine() {
            if (defaultEngine != null) return
            val resolved: EngineType? = when {
                enginesProvider.containsKey(DefaultEngineType) -> DefaultEngineType
                enginesProvider.isNotEmpty() -> enginesProvider.iterator().next().key
                else -> null
            }
            if (defaultEngine == null) {
                defaultEngine = resolved
            }
        }

        override val name: String
            get() = throw EngineMultikException("For a companion object, the name is undefined.")

        override val type: EngineType
            get() = throw EngineMultikException("For a companion object, the type is undefined.")

        internal fun getDefaultEngine(): String? {
            var engine = defaultEngine
            if (engine == null) {
                loadEngine()
                engine = defaultEngine
            }
            return engine?.name
        }

        internal fun setDefaultEngine(type: EngineType) {
            if (!enginesProvider.containsKey(type)) throw EngineMultikException("This type of engine is not available.")
            defaultEngine = type
        }

        override fun getMath(): Math {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Math implementation.")
            var engine = defaultEngine
            if (engine == null) {
                loadEngine()
                engine = defaultEngine
            }
            return enginesProvider[engine]?.getMath()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getLinAlg(): LinAlg {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide LinAlg implementation.")
            var engine = defaultEngine
            if (engine == null) {
                loadEngine()
                engine = defaultEngine
            }
            return enginesProvider[engine]?.getLinAlg()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getStatistics(): Statistics {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Statistics implementation.")
            var engine = defaultEngine
            if (engine == null) {
                loadEngine()
                engine = defaultEngine
            }
            return enginesProvider[engine]?.getStatistics()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }
    }
}

/**
 * Exception thrown when an engine operation fails — e.g., no engine is available
 * or the requested engine type is not registered.
 */
public class EngineMultikException(message: String) : Exception(message) {
    public constructor() : this("")
}

