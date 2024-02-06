/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api
import kotlin.concurrent.AtomicReference

/**
 * Engine Provider for Kotlin/Native targets.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> = engines.value

/**
 * Saves and initialize engine.
 */
public val engines: AtomicReference<MutableMap<EngineType, Engine>> by lazy {
    AtomicReference(mutableMapOf())
}
