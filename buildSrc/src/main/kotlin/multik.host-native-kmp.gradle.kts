import org.jetbrains.kotlinx.multik.builds.HostDetection
import org.jetbrains.kotlinx.multik.builds.MultikNativeTarget

plugins {
    id("multik.base")
}

kotlin {
    val hostTarget = when (HostDetection.currentTarget) {
        MultikNativeTarget.MACOS_X64 -> macosX64()
        MultikNativeTarget.MACOS_ARM64 -> macosArm64()
        MultikNativeTarget.LINUX_X64 -> linuxX64()
        MultikNativeTarget.MINGW_X64 -> mingwX64()
    }

    project.extra["hostNativeTarget"] = hostTarget
}