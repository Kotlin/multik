/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

public interface EngineFactory {
    public fun getEngine(type: EngineType? = null): Engine
}