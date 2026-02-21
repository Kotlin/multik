@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)

import org.jetbrains.kotlinx.multik.builds.HostDetection
import org.jetbrains.kotlinx.multik.builds.MultikNativeTarget

plugins {
    id("multik.base")
    id("multik.publishing")
}

kotlin {
    // Web targets
    wasmJs {
        browser { testTask { enabled = false } }
        nodejs { testTask { enabled = false } }
        binaries.library()
    }
    js {
        browser { testTask { enabled = false } }
        nodejs { testTask { enabled = false } }
        binaries.library()
    }

    // Host-only desktop native
    val hostTarget = when (HostDetection.currentTarget) {
        MultikNativeTarget.MACOS_X64 -> macosX64()
        MultikNativeTarget.MACOS_ARM64 -> macosArm64()
        MultikNativeTarget.LINUX_X64 -> linuxX64()
        MultikNativeTarget.MINGW_X64 -> mingwX64()
    }

    // iOS targets
    iosArm64()
    iosSimulatorArm64()
    iosX64()

    applyDefaultHierarchyTemplate()

    sourceSets {
        commonMain {
            dependencies {
                api(project(":multik-core"))
                implementation(project(":multik-kotlin"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        jvmMain {
            dependencies {
                api(project(":multik-openblas"))
            }
        }

        val desktopMain by creating {
            dependsOn(commonMain.get())
            dependencies {
                api(project(":multik-openblas"))
            }
        }
        val desktopTest by creating {
            dependsOn(commonTest.get())
        }

        val hostName = HostDetection.currentTarget.targetName
        getByName("${hostName}Main").dependsOn(desktopMain)
        getByName("${hostName}Test").dependsOn(desktopTest)
    }
}