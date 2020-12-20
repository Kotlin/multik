package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jni.NativeEngine

public class DefaultEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return DefaultEngine
    }
}

public object DefaultEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = DefaultEngineType

    init {
        NativeEngine
    }

    override fun getMath(): Math {
        return DefaultMath
    }

    override fun getLinAlg(): LinAlg {
        return DefaultLinAlg
    }

    override fun getStatistics(): Statistics {
        return DefaultStatistics
    }
}