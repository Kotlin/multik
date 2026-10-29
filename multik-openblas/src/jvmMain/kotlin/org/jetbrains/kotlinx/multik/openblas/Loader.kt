package org.jetbrains.kotlinx.multik.openblas

import java.io.FileNotFoundException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

internal actual fun libLoader(name: String): Loader = JvmLoader(name)

internal class JvmLoader(private val name: String) : Loader {
    override val isLoaded: Boolean
        get() = _isLoaded

    @Volatile
    private var _isLoaded: Boolean = false

    private val lock = Any()

    private val os: String
        get() {
            val osProperty: String = System.getProperty("os.name").lowercase()
            return when {
                osProperty.contains("mac") -> "macos"
                osProperty.contains("nux") -> "linux"
                osProperty.contains("win") -> "mingw"
                else -> error("Unsupported operating system: $osProperty")
            }
        }

    private val arch: String
        get() = when (val arch: String = System.getProperty("os.arch").lowercase()) {
            "amd64", "x86_64", "x86-64", "x64" -> "X64"
            "arm64", "aarch64", "armv8" -> "Arm64"
            else -> error("Unsupported architecture: $arch")
        }

    private val libraryDir: Path by lazy {
        val path = Files.createTempDirectory("jni_multik")
        path.toFile().deleteOnExit()
        path
    }


    // Resolved lazily: on an unsupported platform os/arch throw, and load() reports that as a failed load.
    private val nativeNameLib by lazy {
        buildString {
            if (os == "mingw")
                append("lib")
            append(name)
            append('-')
            append(os)
            append(arch)
        }
    }

    private val prefixPath by lazy {
        when (os) {
            "linux" -> "lib/linuxX64/"
            "macos" -> "lib/macosArm64/"
            else -> "lib/mingwX64/"
        }
    }

    override fun load(): Boolean {
        synchronized(lock) {
            if (_isLoaded) return true

            var libraryPath: Path? = null

            return try {
                val resource = System.mapLibraryName(nativeNameLib)
                val inputStream = Loader::class.java.classLoader.getResourceAsStream("$prefixPath$resource")
                if (inputStream != null) {
                    libraryPath = libraryDir.resolve(resource)
                    Files.copy(inputStream, libraryPath!!, StandardCopyOption.REPLACE_EXISTING)
                    System.load(libraryPath.toString())
                } else {
                    System.loadLibrary(nativeNameLib)
                }
                _isLoaded = true
                true
            } catch (e: FileNotFoundException) {
                println("Library file not found: ${e.message}")
                false
            } catch (e: SecurityException) {
                println("No access to the library file: ${e.message}")
                false
            } catch (e: Throwable) {
                println("Unknown error: ${e.message}")
                libraryPath?.toFile()?.delete()
                false
            }
        }
    }

    override fun manualLoad(javaPath: String?): Boolean {
        synchronized(lock) {
            if (_isLoaded) return true

            return try {
                if (javaPath.isNullOrEmpty()) {
                    System.loadLibrary(nativeNameLib)
                } else {
                    val libFullName = System.mapLibraryName(nativeNameLib)
                    val fullPath = if (os == "win") "$javaPath\\$libFullName" else "$javaPath/$libFullName"
                    System.load(fullPath)
                }
                _isLoaded = true
                true
            } catch (e: FileNotFoundException) {
                println("Library file not found: ${e.message}")
                false
            } catch (e: SecurityException) {
                println("No access to the library file: ${e.message}")
                false
            } catch (e: Throwable) {
                println("Unknown error: ${e.message}")
                false
            }
        }
    }
}
