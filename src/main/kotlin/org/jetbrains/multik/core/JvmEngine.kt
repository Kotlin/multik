package org.jetbrains.multik.core

import org.jetbrains.multik.api.*


public class JvmEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return JvmEngine()
    }
}

public class JvmEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = EngineType.JVM

    override fun getMath(): Math {
        return JvmMath
    }

    override fun getLinAlg(): LinAlg {
        return JvmLinAlg
    }
}