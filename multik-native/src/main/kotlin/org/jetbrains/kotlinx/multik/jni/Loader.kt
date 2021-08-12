/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

internal class Loader(private val name: String) {

    var loading: Boolean = false
        private set

    private val os: String by lazy {
        val osProperty: String = System.getProperty("os.name").lowercase()
        when {
            osProperty.contains("mac") -> "darwin"
            osProperty.contains("nux") -> "linux"
            osProperty.contains("win") -> "win"
            else -> error("Unsupported operating system: $osProperty")
        }
    }

    // TODO(add different arch)
    private val arch: String
        get() = when (val arch: String = System.getProperty("os.arch")) {
            "amd64", "x86_64" -> "x86_64"
            else -> error("Unsupported architecture: $arch")
        }

    private val libraryDir: Path by lazy {
        val path = Files.createTempDirectory("jni_multik")
        path.toFile().deleteOnExit()
        path
    }


    private val nativeNameLib = buildString {
        append(name)
//        append("$os.")
//        append(arch)
    }

    fun load(): Boolean {
        val resource = System.mapLibraryName(nativeNameLib)
        val inputStream = Loader::class.java.getResourceAsStream("/$resource")
        var libraryPath: Path? = null
        return try {
            if (inputStream != null) {
                libraryPath = libraryDir.resolve(resource)
                Files.copy(inputStream, libraryPath!!, StandardCopyOption.REPLACE_EXISTING)
                System.load(libraryPath.toString())
            } else {
                System.loadLibrary(nativeNameLib)
            }
            loading = true
            true
        } catch (e: Throwable) {
            libraryPath?.toFile()?.delete()
            throw e
        }
    }

    fun manualLoad(javaPath: String? = null): Boolean {
        if (javaPath.isNullOrEmpty()) {
            System.loadLibrary(name)
        } else {
            val libFullName = System.mapLibraryName(name)
            val fullPath = if (os == "win") "$javaPath\\$libFullName" else "$javaPath/$libFullName"
            System.load(fullPath)
        }
        loading = true
        return true
    }
}