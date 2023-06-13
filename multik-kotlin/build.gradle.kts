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

    wasm {
        browser {
            testTask {
                /*
                    https://youtrack.jetbrains.com/issue/KT-56633
                    https://youtrack.jetbrains.com/issue/KT-56159
                 */
                this.enabled = false // fixed in 1.9.0/1.9.20
            }
        }
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
        val wasmMain by getting {
            dependsOn(commonMain)
        }
        val jsMain by getting {
            dependsOn(commonMain)
        }
        val nativeMain by creating {
            dependsOn(commonMain)
        }
        names.forEach { n ->
            if (n.contains("X64Main") || n.contains("Arm64Main")) {
                this@sourceSets.getByName(n).apply {
                    dependsOn(nativeMain)
                }
            }
        }
    }
}

