package org.jetbrains.kotlinx.multik.openblas

internal actual fun libLoader(name: String): Loader = AndroidLoader(name)

/**
 * Loads the JNI library that the AAR ships in `jni/<abi>/`.
 * Android installs it together with the app, so the library is loaded by name and nothing is extracted.
 */
internal class AndroidLoader(private val name: String) : Loader {
    override val isLoaded: Boolean
        get() = _isLoaded

    @Volatile
    private var _isLoaded: Boolean = false

    private val lock = Any()

    // Resolved lazily: on an unsupported ABI this throws, and load() reports that as a failed load.
    private val nativeNameLib by lazy {
        val arch = when (val arch: String = System.getProperty("os.arch").lowercase()) {
            "aarch64", "arm64" -> "Arm64"
            "x86_64", "amd64" -> "X64"
            else -> error("Unsupported architecture: $arch")
        }
        "$name-android$arch"
    }

    override fun load(): Boolean = manualLoad(null)

    override fun manualLoad(javaPath: String?): Boolean {
        synchronized(lock) {
            if (_isLoaded) return true

            return try {
                if (javaPath.isNullOrEmpty()) {
                    System.loadLibrary(nativeNameLib)
                } else {
                    System.load("$javaPath/${System.mapLibraryName(nativeNameLib)}")
                }
                _isLoaded = true
                true
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
