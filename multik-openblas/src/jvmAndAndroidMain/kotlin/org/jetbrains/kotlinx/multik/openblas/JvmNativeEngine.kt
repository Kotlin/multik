package org.jetbrains.kotlinx.multik.openblas

import org.jetbrains.kotlinx.multik.api.EngineMultikException
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.math.Math
import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.openblas.linalg.NativeLinAlg
import org.jetbrains.kotlinx.multik.openblas.math.NativeMath
import org.jetbrains.kotlinx.multik.openblas.stat.NativeStatistics

public class JvmNativeEngine: NativeEngine() {
    private val loader: Loader by lazy { libLoader("multik_jni") }

    override fun getMath(): Math {
        ensureLoaded()
        return NativeMath
    }

    override fun getLinAlg(): LinAlg {
        ensureLoaded()
        return NativeLinAlg
    }

    override fun getStatistics(): Statistics {
        ensureLoaded()
        return NativeStatistics
    }

    private fun ensureLoaded() {
        if (!loader.isLoaded && !loader.load()) {
            throw EngineMultikException(
                "Failed to load the multik_jni native library on " +
                    "${System.getProperty("os.name")} ${System.getProperty("os.arch")}."
            )
        }
    }
}
