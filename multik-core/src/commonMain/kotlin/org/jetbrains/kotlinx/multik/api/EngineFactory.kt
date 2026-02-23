/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

/**
 * Factory interface for creating and retrieving [Engine] instances.
 *
 * Implementations of this interface are responsible for providing access to computational engines
 * that power Multik's mathematical operations, linear algebra, and statistics.
 *
 * This is a simple factory pattern where implementations return appropriate [Engine] instances
 * based on the requested [EngineType]. The actual engine discovery mechanism (e.g., ServiceLoader
 * on JVM, eager initialization on Native/JS/WASM) is handled by [enginesProvider], not by this interface.
 *
 * @see Engine
 * @see EngineType
 * @see enginesProvider
 */
public interface EngineFactory {
    /**
     * Returns an [Engine] instance of the specified type.
     *
     * @param type the desired engine type, or null to request the default behavior (implementation-specific).
     * @return an Engine instance matching the requested type.
     * @throws IllegalStateException if the requested engine type is not available or not supported.
     */
    public fun getEngine(type: EngineType? = null): Engine
}