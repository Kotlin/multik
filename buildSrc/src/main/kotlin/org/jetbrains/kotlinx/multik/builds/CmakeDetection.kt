package org.jetbrains.kotlinx.multik.builds

import java.io.File

/**
 * Resolves an absolute path to the `cmake` executable.
 *
 * The `cmake` Gradle tasks run as `Exec` and inherit the PATH of whatever process launched Gradle.
 * GUI-launched IDEs on macOS don't pick up Homebrew's `/opt/homebrew/bin`, so `commandLine("cmake", ...)`
 * can fail with "Cannot run program 'cmake'" even when the binary is installed. Resolving to an
 * absolute path at configuration time sidesteps PATH inheritance entirely.
 */
object CmakeDetection {

    /** Absolute path to `cmake`, or plain `"cmake"` as a last resort. */
    val executable: String by lazy {
        System.getenv("CMAKE")?.let(::File)?.takeIf { it.isFile && it.canExecute() }?.absolutePath
            ?: findOnPath()
            ?: commonInstallPaths.map(::File).firstOrNull { it.isFile && it.canExecute() }?.absolutePath
            ?: "cmake"
    }

    private val commonInstallPaths: List<String> = when {
        HostDetection.isMacosArm64 -> listOf("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake")
        HostDetection.isLinux -> listOf("/usr/bin/cmake", "/usr/local/bin/cmake")
        HostDetection.isWindows -> listOf(
            "C:\\Program Files\\CMake\\bin\\cmake.exe",
            "C:\\Program Files (x86)\\CMake\\bin\\cmake.exe"
        )
        else -> emptyList()
    }

    private fun findOnPath(): String? {
        val path = System.getenv("PATH") ?: return null
        val separator = File.pathSeparator
        val exeNames = if (HostDetection.isWindows) listOf("cmake.exe", "cmake") else listOf("cmake")
        for (dir in path.split(separator)) {
            if (dir.isEmpty()) continue
            for (name in exeNames) {
                val candidate = File(dir, name)
                if (candidate.isFile && candidate.canExecute()) return candidate.absolutePath
            }
        }
        return null
    }
}
