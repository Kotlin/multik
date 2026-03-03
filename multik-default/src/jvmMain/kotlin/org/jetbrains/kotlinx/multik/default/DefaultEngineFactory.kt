package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.DefaultEngineType
import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineFactory
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.kotlin.KEEngine
import org.jetbrains.kotlinx.multik.openblas.JvmNativeEngine

internal actual object DefaultEngineFactory : EngineFactory {
    actual override fun getEngine(type: EngineType?): Engine =
        when (type) {
            null -> error("Pass NativeEngineType of KEEngineType")
            KEEngineType -> KEEngine()
            NativeEngineType -> {
                if (supportNative)
                    JvmNativeEngine()
                else
                    KEEngine()
            }
            DefaultEngineType -> error("Default Engine doesn't link to Default Engine")
        }

    private val supportNative: Boolean by lazy {
        when ("$os-$arch") {
            "macos-X64", "macos-Arm64", "linux-X64", "mingw-X64", "android-Arm64" -> true
            else -> false
        }
    }

    private val os: String
        get() {
            val osProperty: String = System.getProperty("os.name").lowercase()
            return when {
                osProperty.contains("mac") -> "macos"
                System.getProperty("java.vm.name").contains("Dalvik") -> "android"
                osProperty.contains("nux") -> "linux"
                osProperty.contains("win") -> "mingw"
                else -> "Unknown os"
            }
        }

    private val arch: String
        get() = when (val arch: String = System.getProperty("os.arch").lowercase()) {
            "amd64", "x86_64", "x86-64", "x64" -> "X64"
            "arm64", "aarch64", "armv8" -> "Arm64"
            else -> "Unknown arch"
        }
}
