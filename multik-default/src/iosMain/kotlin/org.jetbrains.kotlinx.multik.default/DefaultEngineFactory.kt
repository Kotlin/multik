package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.DefaultEngineType
import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineFactory
import org.jetbrains.kotlinx.multik.api.EngineType
import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.kotlin.KEEngine

internal actual object DefaultEngineFactory : EngineFactory {
    actual override fun getEngine(type: EngineType?): Engine =
        when (type) {
            null, KEEngineType, DefaultEngineType -> KEEngine()
            NativeEngineType -> error("Don't exist native engine for iOS targets")
        }
}
