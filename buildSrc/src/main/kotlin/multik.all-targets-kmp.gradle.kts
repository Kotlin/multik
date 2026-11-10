@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.plugin.getKotlinPluginVersion

plugins {
    id("multik.base")
}

kotlin {
    mingwX64()
    linuxX64()
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

    sourceSets {
        val compilerVersion = project.getKotlinPluginVersion()
        jsMain.dependencies { implementation(kotlin("stdlib", compilerVersion)) }
        wasmJsMain.dependencies { implementation(kotlin("stdlib", compilerVersion)) }
        webTest.dependencies { implementation(kotlin("test", compilerVersion)) }
    }
}
