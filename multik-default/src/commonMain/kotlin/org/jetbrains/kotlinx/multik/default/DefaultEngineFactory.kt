/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default

import org.jetbrains.kotlinx.multik.api.Engine
import org.jetbrains.kotlinx.multik.api.EngineFactory
import org.jetbrains.kotlinx.multik.api.EngineType

internal expect object DefaultEngineFactory: EngineFactory {
    override fun getEngine(type: EngineType?): Engine
}