import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeCompile
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeLink
import org.jetbrains.kotlinx.multik.builds.HostDetection
import org.jetbrains.kotlinx.multik.builds.MultikNativeTarget

plugins {
    id("multik.base")
}

kotlin {
    macosArm64()
    macosX64()
    linuxX64()
    mingwX64()

    val hostTarget = when (HostDetection.currentTarget) {
        MultikNativeTarget.MACOS_X64 -> targets.getByName("macosX64")
        MultikNativeTarget.MACOS_ARM64 -> targets.getByName("macosArm64")
        MultikNativeTarget.LINUX_X64 -> targets.getByName("linuxX64")
        MultikNativeTarget.MINGW_X64 -> targets.getByName("mingwX64")
    }
    project.extra["hostNativeTarget"] = hostTarget
}

// Disable compile, link, and test tasks for non-host native targets.
// CInterop tasks are kept enabled so stubs are generated for IDE support.
// Each CI machine builds and publishes only its own platform.
val hostTargetName = (project.extra["hostNativeTarget"] as KotlinNativeTarget).name
tasks.configureEach {
    if (this is KotlinNativeCompile || this is KotlinNativeLink) {
        val isNonHost = kotlin.targets.withType<KotlinNativeTarget>()
            .matching { it.name != hostTargetName }
            .any { name.contains(it.name, ignoreCase = true) }
        if (isNonHost) enabled = false
    }
}