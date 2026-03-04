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

val hostTargetName = (project.extra["hostNativeTarget"] as KotlinNativeTarget).name
val allNativeTargetNames = kotlin.targets.withType<KotlinNativeTarget>().map { it.name }

// multik.mainHost (Gradle property, default=true):
//   true  → main host: publishes everything except non-host native
//   false → platform runner: publishes only host native klib
val isMainHost = (findProperty("multik.mainHost") as? String)?.toBoolean() ?: true

tasks.configureEach {
    when (this) {
        is KotlinNativeCompile, is KotlinNativeLink -> {
            val isNonHost = allNativeTargetNames
                .filter { it != hostTargetName }
                .any { name.contains(it, ignoreCase = true) }
            if (isNonHost) enabled = false
        }

        is AbstractPublishToMaven, is GenerateModuleMetadata -> {
            if (isMainHost) {
                val isNonHostNative = allNativeTargetNames
                    .filter { it != hostTargetName }
                    .any { name.contains(it, ignoreCase = true) }
                if (isNonHostNative) enabled = false
            } else {
                val isHostNative = name.contains(hostTargetName, ignoreCase = true)
                if (!isHostNative) enabled = false
            }
        }
    }
}
