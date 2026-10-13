package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.DefaultEngineType
import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineFactory
import org.jetbrains.kotlinx.multik.api.EngineMultikException
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.kotlin.KEEngine
import org.jetbrains.kotlinx.multik.openblas.JvmNativeEngine

internal actual object DefaultEngineFactory : EngineFactory {
    actual override fun getEngine(type: EngineType?): Engine =
        when (type) {
            null -> throw IllegalArgumentException("Engine type is required: pass NativeEngineType or KEEngineType.")
            KEEngineType -> KEEngine()
            NativeEngineType -> nativeEngine ?: KEEngine()
            DefaultEngineType -> error("Default Engine doesn't link to Default Engine")
        }

    // Native support is decided by loading the library, not by os/arch. Where the jar or AAR has no library
    // for the platform (another ABI, Android unit tests on the host JVM), the Kotlin engine is used instead.
    private val nativeEngine: Engine? by lazy {
        val engine = JvmNativeEngine()
        try {
            engine.getMath() // loads the library
            engine
        } catch (e: EngineMultikException) {
            null
        }
    }
}
