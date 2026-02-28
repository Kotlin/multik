package org.jetbrains.kotlinx.multik.builds

/**
 * Detects Homebrew-installed GCC on macOS.
 * Returns null for all properties on non-macOS platforms or if Homebrew GCC is not installed.
 */
object HomebrewGccDetection {

    /** Homebrew GCC prefix (e.g. `/opt/homebrew/opt/gcc`), or null if not installed. */
    val prefix: String? by lazy {
        if (!isMacos) return@lazy null
        try {
            val process = ProcessBuilder("brew", "--prefix", "gcc").start()
            val output = process.inputStream.bufferedReader().readText().trim()
            val exitCode = process.waitFor()
            if (exitCode == 0 && output.isNotEmpty() && java.io.File(output).isDirectory) output else null
        } catch (_: Exception) {
            null
        }
    }

    /** Major version number of the installed GCC (e.g. 15), or null. */
    val majorVersion: Int? by lazy {
        val binDir = prefix?.let { java.io.File(it, "bin") } ?: return@lazy null
        binDir.listFiles()
            ?.mapNotNull { f ->
                val match = Regex("""^gcc-(\d+)$""").matchEntire(f.name)
                match?.groupValues?.get(1)?.toIntOrNull()
            }
            ?.maxOrNull()
    }

    /** Path to versioned `gcc-N` binary, or null. */
    val cCompiler: String? by lazy {
        val v = majorVersion ?: return@lazy null
        val path = "$prefix/bin/gcc-$v"
        if (java.io.File(path).canExecute()) path else null
    }

    /** Path to versioned `g++-N` binary, or null. */
    val cxxCompiler: String? by lazy {
        val v = majorVersion ?: return@lazy null
        val path = "$prefix/bin/g++-$v"
        if (java.io.File(path).canExecute()) path else null
    }

    /** GCC library path (e.g. `<prefix>/lib/gcc/current`), or null. */
    val libPath: String? by lazy {
        val current = prefix?.let { java.io.File(it, "lib/gcc/current") }
        if (current != null && current.isDirectory) current.absolutePath else null
    }

    private val isMacos: Boolean get() = HostDetection.isMacosX64 || HostDetection.isMacosArm64
}
