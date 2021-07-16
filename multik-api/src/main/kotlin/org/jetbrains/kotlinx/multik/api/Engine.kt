/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import java.util.*
import java.util.concurrent.ConcurrentHashMap

public sealed class EngineType(public val name: String)

public object DefaultEngineType : EngineType("DEFAULT")

public object JvmEngineType : EngineType("JVM")

public object NativeEngineType : EngineType("NATIVE")

public object CudaEngineType : EngineType("CUDA")

/**
 * This class gives access to different implementations of [LinAlg], [Math], [Statistics].
 * When initializing [Multik], it loads engines, by default `DEFAULT` implementation is used.
 */
public abstract class Engine {

    protected abstract val name: String

    protected abstract val type: EngineType

    protected val engines: MutableMap<EngineType, Engine> = ConcurrentHashMap<EngineType, Engine>()

    protected var defaultEngine: EngineType? = null

    protected fun loadEngine() {
        val loaders: ServiceLoader<EngineProvider> = ServiceLoader.load(EngineProvider::class.java)
        for (engineProvider in loaders) {
            val engine = engineProvider.getEngine()
            if (engine != null) {
                engines[engine.type] = engine
            }
        }

        defaultEngine = when {
            engines.containsKey(DefaultEngineType) -> DefaultEngineType
            engines.isNotEmpty() -> engines.iterator().next().key
            else -> null
        }
    }

    /**
     * Returns [Math] implementation.
     */
    public abstract fun getMath(): Math

    /**
     * Returns [LinAlg] implementation.
     */
    public abstract fun getLinAlg(): LinAlg

    /**
     * Returns [Statistics] implementation.
     */
    public abstract fun getStatistics(): Statistics

    internal companion object : Engine() {

        init {
            loadEngine()
        }

        override val name: String
            get() = throw EngineMultikException("For a companion object, the name is undefined.")

        override val type: EngineType
            get() = throw EngineMultikException("For a companion object, the type is undefined.")

        internal fun getDefaultEngine(): String? = defaultEngine?.name

        internal fun setDefaultEngine(type: EngineType) {
            if (!engines.containsKey(type)) throw EngineMultikException("This type of engine is not available.")
            defaultEngine = type
        }

        override fun getMath(): Math {
            if (engines.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Math implementation.")
            return engines[defaultEngine]?.getMath()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getLinAlg(): LinAlg {
            if (engines.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide LinAlg implementation.")
            return engines[defaultEngine]?.getLinAlg()
                ?: throw throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getStatistics(): Statistics {
            if (engines.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Statistics implementation.")
            return engines[defaultEngine]?.getStatistics()
                ?: throw throw EngineMultikException("The used engine type is not defined.")
        }
    }
}

public interface EngineProvider {
    public fun getEngine(): Engine?
}

public class EngineMultikException(message: String) : Exception(message) {
    public constructor() : this("")
}