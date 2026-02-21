@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)

plugins {
    id("multik.base")
}

kotlin {
    mingwX64()
    linuxX64()
    macosX64()
    macosArm64()

    iosArm64()
    iosSimulatorArm64()
    iosX64()

    wasmJs {
        browser {
            testTask { enabled = false }
        }
        nodejs {
            testTask { enabled = false }
        }
        binaries.library()
    }

    js {
        browser {
            testTask { enabled = false }
        }
        nodejs {
            testTask { enabled = false }
        }
        binaries.library()
    }
}