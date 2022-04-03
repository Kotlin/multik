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
        compilations.getByName("main") {
            cinterops {
                val libmultik by creating {
                    defFile(project.file(("src/cinterop/libmultik.def")))
                    packageName("org.jetbrains.kotlinx.multik.jni")
//                    extraOpts("-Xsource-compiler-option", "-I/Users/pavel.gorgulov/Projects/main_project/multik/multik-native/multik_jni/src/main/headers")
//                    extraOpts("-Xsource-compiler-option", "-DONLY_C_LOCALE=1")
                    extraOpts("-Xsource-compiler-option", "-std=c++17")
//                    compilerOpts("-I/Users/pavel.gorgulov/Projects/main_project/multik/multik-native/multik_jni/src/main/headers")
                    includeDirs("/Users/pavel.gorgulov/Projects/main_project/multik/multik-native/multik_jni/src/main/headers")
//                    includeDirs.allHeaders("path")
                }
            }
        }
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
            if (n.contains("X64Main") || n.contains("Arm64Main")) {
                this@sourceSets.getByName(n).apply {
                    dependsOn(nativeMain)
                }
            }
        }
    }
}

