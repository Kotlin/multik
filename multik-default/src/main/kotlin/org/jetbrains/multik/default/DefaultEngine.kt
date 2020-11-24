package org.jetbrains.multik.default

import org.jetbrains.multik.api.*
import org.jetbrains.multik.jni.NativeEngine

public class DefaultEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return DefaultEngine
    }
}

public object DefaultEngine : Engine() {

    init {
        NativeEngine
    }

    override val name: String
        get() = type.name

    override val type: EngineType
        get() = DefaultEngineType

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