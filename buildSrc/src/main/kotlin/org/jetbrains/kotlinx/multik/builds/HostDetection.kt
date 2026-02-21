package org.jetbrains.kotlinx.multik.builds

import org.gradle.api.GradleException

enum class MultikNativeTarget(val targetName: String) {
    MACOS_X64("macosX64"),
    MACOS_ARM64("macosArm64"),
    LINUX_X64("linuxX64"),
    MINGW_X64("mingwX64"),
}

object HostDetection {
    val hostOs: String = System.getProperty("os.name")
    val hostArch: String = System.getProperty("os.arch")

    val isMacosX64: Boolean get() = hostOs == "Mac OS X" && hostArch == "x86_64"
    val isMacosArm64: Boolean get() = hostOs == "Mac OS X" && hostArch == "aarch64"
    val isLinux: Boolean get() = hostOs == "Linux"
    val isWindows: Boolean get() = hostOs.startsWith("Windows")

    val currentTarget: MultikNativeTarget
        get() = when {
            isMacosX64 -> MultikNativeTarget.MACOS_X64
            isMacosArm64 -> MultikNativeTarget.MACOS_ARM64
            isLinux -> MultikNativeTarget.LINUX_X64
            isWindows -> MultikNativeTarget.MINGW_X64
            else -> throw GradleException(
                """
                Failed to detect platform. Supported platforms:
                    macosX64, macosArm64, linuxX64, mingwX64
                """.trimIndent()
            )
        }

    /** Target OS string for CMake, overridable via TARGET_OS env var. */
    val targetOSForCMake: String
        get() = System.getenv("TARGET_OS") ?: currentTarget.targetName
}