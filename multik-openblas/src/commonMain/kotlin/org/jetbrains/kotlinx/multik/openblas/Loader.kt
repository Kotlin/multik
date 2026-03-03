package org.jetbrains.kotlinx.multik.openblas

internal expect fun libLoader(name: String): Loader

internal interface Loader {
    val isLoaded: Boolean

    fun load(): Boolean

    fun manualLoad(javaPath: String? = null): Boolean
}
