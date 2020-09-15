package org.jetbrains.multik.jni

import org.jetbrains.multik.api.*


public class NativeEngineProvider : EngineProvider {
    override fun getEngine(): Engine? {
        return NativeEngine()
    }
}

public class NativeEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = EngineType.NATIVE

    override fun getMath(): Math {
        return NativeMath
    }

    override fun getLinAlg(): LinAlg {
        return NativeLinAlg
    }
}