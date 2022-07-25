package org.jetbrains.kotlinx.multik.openblas

public class JvmNativeEngine: NativeEngine() {
    private val loader: Loader by lazy { libLoader("multik_jni") }

    init {
        if(!loader.loading) loader.load()
    }
}