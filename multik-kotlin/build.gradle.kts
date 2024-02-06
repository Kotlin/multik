@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.targets.js.dsl.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform")
}


kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        testRuns["test"].executionTask.configure {
            useJUnit()
        }
    }
    mingwX64()
    linuxX64()
    macosX64 {
        binaries {
            framework {
                baseName = "multik-kotlin"
            }
        }
    }
    macosArm64 {
        binaries {
            framework {
                baseName = "multik-kotlin"
            }
        }
    }
    iosArm64 {
        binaries {
            framework {
                baseName = "multik-kotlin"
            }
        }
    }
    iosSimulatorArm64 {
        binaries {
            framework {
                baseName = "multik-kotlin"
            }
        }
    }
    iosX64 {
        binaries {
            framework {
                baseName = "multik-kotlin"
            }
        }
    }

    wasmJs {
        browser {
            testTask {
                enabled = false
            }
        }
        nodejs {
            testTask {
                enabled = false
            }
        }
        d8()
    }

    js(IR) {
        val timeoutMs = "1000000"
        browser {
            testTask {
                useMocha {
                    timeout = timeoutMs
                }
            }
        }
        nodejs {
            testTask {
                useMocha {
                    timeout = timeoutMs
                }
            }
        }
    }

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        binaries.getTest("debug").apply {
            debuggable = false
            optimized = true
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":multik-core"))
            }
        }
        val commonTest by getting {
            dependencies {
                api(project(":multik-core"))
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("reflect"))
            }
        }
    }
}
