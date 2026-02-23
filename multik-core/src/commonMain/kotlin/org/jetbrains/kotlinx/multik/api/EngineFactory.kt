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
 * Engine selection is platform-specific:
 * - On JVM: Uses ServiceLoader to discover available engines
 * - On Native/JS/WASM: Uses eager initialization
 *
 * @see Engine
 * @see EngineType
 */
public interface EngineFactory {
    /**
     * Returns an [Engine] instance of the specified type.
     *
     * @param type the desired engine type, or null to use the default engine for the platform.
     * @return an Engine instance matching the requested type.
     * @throws EngineMultikException if the requested engine type is not available.
     */
    public fun getEngine(type: EngineType? = null): Engine
}