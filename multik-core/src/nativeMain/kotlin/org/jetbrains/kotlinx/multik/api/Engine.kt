package org.jetbrains.kotlinx.multik.api

import kotlin.concurrent.AtomicReference

/**
 * Engine Provider for Kotlin/Native targets.
 */
public actual fun enginesProvider(): Map<EngineType, Engine> = engines.value

/**
 * Thread-safe registry where Native engine factories store their [Engine] instances during initialization.
 *
 * Engines register themselves via `@EagerInitialization` objects. [enginesProvider] returns the current snapshot.
 */
public val engines: AtomicReference<MutableMap<EngineType, Engine>> by lazy {
    AtomicReference(mutableMapOf())
}
