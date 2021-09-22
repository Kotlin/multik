package org.jetbrains.kotlinx.multik.api


public actual val enginesProvider : Set<Engine> =
    mutableSetOf()


public actual fun initEnginesProvider(engines: List<Engine>) {
    (enginesProvider as MutableSet).addAll(engines)
}