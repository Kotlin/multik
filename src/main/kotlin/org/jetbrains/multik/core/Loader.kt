package org.jetbrains.multik.core

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
        //TODO (Delete os, add arch)
//        append("$os.")
//        append(arch)
    }

    fun load(): Boolean {
        val resource = System.mapLibraryName(nativeNameLib)
        val inputStream = Loader::class.java.getResourceAsStream("/$resource")
        return try {
            if (inputStream != null) {
                val libraryPath = libraryDir.resolve(resource)
                Files.copy(inputStream, libraryPath, StandardCopyOption.REPLACE_EXISTING)
                System.load(libraryPath.toString())
            } else {
                System.loadLibrary(nativeNameLib)
            }
            true
        } catch (e: Throwable) {
            false
        }
    }
}