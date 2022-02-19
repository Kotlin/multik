/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

internal expect fun libLoader(name: String): Loader

internal interface Loader {
    val loading: Boolean

    fun load(): Boolean

    fun manualLoad(javaPath: String? = null): Boolean
}
