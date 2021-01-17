/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

internal class Loader(private val name: String) {

    // TODO(add linux and win)
    private val os: String
        get() = when (val os: String = System.getProperty("os.name")) {
            "Mac OS X" -> "darwin"
            else -> error("Unsupported operating system: $os")
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
            true
        } catch (e: Throwable) {
            libraryPath?.toFile()?.delete()
            throw e
        }
    }
}