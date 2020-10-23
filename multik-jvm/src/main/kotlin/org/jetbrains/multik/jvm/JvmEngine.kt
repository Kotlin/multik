package org.jetbrains.multik.jvm

import org.jetbrains.multik.api.*


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
}