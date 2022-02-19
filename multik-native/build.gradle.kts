/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    kotlin("multiplatform")
}

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        withJava()
        testRuns["test"].executionTask.configure {
            useJUnit()
        }
    }
    mingwX64()
    linuxX64()
    macosX64 {
        binaries {
            framework {
                baseName = "multik-native"
            }
        }
    }
    macosArm64 {
        binaries {
            framework {
                baseName = "multik-native"
            }
        }
    }
    iosArm64 {
        binaries {
            framework {
                baseName = "multik-native"
            }
        }
    }

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        binaries.all {
            freeCompilerArgs = freeCompilerArgs + "-Xallocator=mimalloc"
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
        val jvmMain by getting
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

