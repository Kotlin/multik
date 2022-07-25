/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

/**
 * Engine Provider for JS.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> = enginesStore

/**
 * Saves and initialize engine.
 */
public val enginesStore: MutableMap<EngineType, Engine> = mutableMapOf()
