package org.jetbrains.kotlinx.multik.api

/**
 * Factory interface for creating [Engine] instances.
 *
 * Implemented by engine modules (e.g., `DefaultEngineFactory`) to provide their engine
 * when requested by [Multik].
 */
public interface EngineFactory {
    /**
     * Returns an [Engine] instance, optionally of the specified [type].
     *
     * @param type the desired [EngineType], or `null` for the default.
     */
    public fun getEngine(type: EngineType? = null): Engine
}
