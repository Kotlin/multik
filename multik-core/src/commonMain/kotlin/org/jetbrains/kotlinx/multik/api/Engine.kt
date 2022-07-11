/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math

public sealed class EngineType(public val name: String)

public object DefaultEngineType : EngineType("DEFAULT")

public object JvmEngineType : EngineType("JVM")

public object NativeEngineType : EngineType("NATIVE")

public expect fun enginesProvider(): Map<EngineType, Engine>

/**
 * This class gives access to different implementations of [LinAlg], [Math], [Statistics].
 * When initializing [Multik], it loads engines, by default `DEFAULT` implementation is used.
 */
public abstract class Engine {

    protected abstract val name: String

    public abstract val type: EngineType

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

        private val enginesProvider: Map<EngineType, Engine> = enginesProvider()

        public var defaultEngine: EngineType? = null

        public fun loadEngine() {
            defaultEngine = when {
                enginesProvider.containsKey(DefaultEngineType) -> DefaultEngineType
                enginesProvider.isNotEmpty() -> enginesProvider.iterator().next().key
                else -> null
            }
        }

        override val name: String
            get() = throw EngineMultikException("For a companion object, the name is undefined.")

        override val type: EngineType
            get() = throw EngineMultikException("For a companion object, the type is undefined.")

        internal fun getDefaultEngine(): String? = defaultEngine?.name ?: loadEngine().let { defaultEngine?.name }

        internal fun setDefaultEngine(type: EngineType) {
            if (!enginesProvider.containsKey(type)) throw EngineMultikException("This type of engine is not available.")
            defaultEngine = type
        }

        override fun getMath(): Math {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Math implementation.")
            if (defaultEngine == null) loadEngine()
            return enginesProvider[defaultEngine]?.getMath()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getLinAlg(): LinAlg {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide LinAlg implementation.")
            if (defaultEngine == null) loadEngine()
            return enginesProvider[defaultEngine]?.getLinAlg()
                ?: throw throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getStatistics(): Statistics {
            if (enginesProvider.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Statistics implementation.")
            if (defaultEngine == null) loadEngine()
            return enginesProvider[defaultEngine]?.getStatistics()
                ?: throw throw EngineMultikException("The used engine type is not defined.")
        }
    }
}

public class EngineMultikException(message: String) : Exception(message) {
    public constructor() : this("")
}

