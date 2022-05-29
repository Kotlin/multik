/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    kotlin("multiplatform")
}


kotlin {
    explicitApi()
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        testRuns["test"].executionTask.configure {
            useJUnit()
        }
//        withJava()
//        val jvmJar by tasks.getting(org.gradle.jvm.tasks.Jar::class) {
//            doFirst {
//                manifest {
//                    attributes["Implementation-Title"] = project.name
//                    attributes["Implementation-Version"] = project.version
//                }
//                from(configurations.getByName("runtimeClasspath").map { if (it.isDirectory) it else zipTree(it) })
//            }
//        }
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

    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    val hostTarget = when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64 {
            binaries { framework { baseName = "multik-default" } }
        }
        hostOs == "Mac OS X" && hostArch == "aarch64" -> macosArm64 {
            binaries { framework { baseName = "multik-default" } }
        }
        hostOs == "Linux" -> linuxX64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }
    iosArm64 {
        binaries {
            framework {
                baseName = "multik-default"
            }
        }
    }
    iosSimulatorArm64 {
        binaries {
            framework {
                baseName = "multik-default"
            }
        }
    }
    iosX64 {
        binaries {
            framework {
                baseName = "multik-default"
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
                implementation(project(":multik-jvm"))
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(project(":multik-native"))
                implementation(kotlin("reflect"))
            }
        }
        val jsMain by getting {
            dependsOn(commonMain)
        }
        val iosMain by creating {
            dependsOn(commonMain)
        }
        names.forEach { name ->
            if (name.contains("iosArm64Main") ||
                name.contains("iosSimulatorArm64Main") ||
                name.contains("iosX64Main")
            ) {
                this@sourceSets.getByName(name).apply {
                    dependsOn(iosMain)
                }
            }
        }

        val nativeMain by creating {
            dependsOn(commonMain)
            dependencies {
                implementation(project(":multik-native"))
            }
        }

        names.forEach { name ->
            if (name.contains("macos") || name.contains("linux") || name.contains("mingw")
            ) {
                this@sourceSets.getByName(name).apply {
                    dependsOn(nativeMain)
                }
            }
        }
    }
}

