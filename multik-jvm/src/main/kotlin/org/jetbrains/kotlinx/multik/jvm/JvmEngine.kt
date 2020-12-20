package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*


public class JvmEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return JvmEngine
    }
}

public object JvmEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = JvmEngineType

    override fun getMath(): Math {
        return JvmMath
    }

    override fun getLinAlg(): LinAlg {
        return JvmLinAlg
    }

    override fun getStatistics(): Statistics {
        return JvmStatistics
    }
}