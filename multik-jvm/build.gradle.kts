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
        withJava()
    }
    mingwX64()
    linuxX64()
    macosX64() {
        binaries {
            framework {
                baseName = "multik-jvm"
            }
        }
    }
    macosArm64() {
        binaries {
            framework {
                baseName = "multik-jvm"
            }
        }
    }
    iosArm64 {
        binaries {
            framework {
                baseName = "multik-jvm"
            }
        }
    }
    iosSimulatorArm64() {
        binaries {
            framework {
                baseName = "multik-jvm"
            }
        }
    }
    iosX64 {
        binaries {
            framework {
                baseName = "multik-jvm"
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
        binaries.all {
            freeCompilerArgs = freeCompilerArgs + "-Xallocator=mimalloc"
            freeCompilerArgs = freeCompilerArgs + "-Xgc=cms"
        }

        binaries.getTest("debug").apply {
            debuggable = false
            optimized = true
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":multik-api"))
            }
        }
        val commonTest by getting {
            dependencies {
                api(project(":multik-api"))
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("reflect"))
            }
        }
        val jsMain by getting {
            dependsOn(commonMain)
        }
        val nativeMain by creating {
            dependsOn(commonMain)
        }
        names.forEach { n ->
            if (n.contains("X64Main") || n.contains("Arm64Main")){
                this@sourceSets.getByName(n).apply{
                    dependsOn(nativeMain)
                }
            }
        }
    }
}

