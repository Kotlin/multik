package org.jetbrains.kotlinx.multik.openblas

internal actual fun libLoader(name: String): Loader = NativeLoader()

internal class NativeLoader(override val isLoaded: Boolean = true) : Loader {
    override fun load(): Boolean = true

    override fun manualLoad(javaPath: String?): Boolean = true
}