/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeCompilation
import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.konan.target.Architecture.ARM64
import org.jetbrains.kotlin.konan.target.Architecture.X64
import org.jetbrains.kotlin.konan.target.Family.*

plugins {
    kotlin("multiplatform")
}

apply(from = "$rootDir/gradle/multik_jni-cmake.gradle")

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
    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    val hostTarget = when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64 {
            binaries {
                framework { baseName = "multik-native" }
            }
        }
        hostOs == "Mac OS X" && hostArch == "aarch64" -> {
            macosArm64 {
                binaries {
                    framework { baseName = "multik-native" }
                }
            }
            iosArm64 {
                        binaries {
            framework { baseName = "multik-native" }
        }
            }
        }
        hostOs == "Linux" -> linuxX64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        compilations.getByName("main") {
            cinterops {
                when {
                    konanTarget.family == OSX && konanTarget.architecture == X64 -> settingCinteropMultik()
                    konanTarget.family == LINUX && konanTarget.architecture == X64 -> settingCinteropMultik()
                    konanTarget.family == MINGW && konanTarget.architecture == X64 -> settingCinteropMultik()
                    konanTarget.family == OSX && konanTarget.architecture == ARM64 -> settingCinteropMultik()
                    konanTarget.family == IOS && konanTarget.architecture == ARM64 -> settingCinteropMultik()
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

tasks.withType(CInteropProcess::class.java).configureEach { dependsOn("build_cmake") }

fun KotlinNativeCompilation.settingCinteropMultik() {
    val libmultik by cinterops.creating {
        val cinteropDir = "${projectDir}/cinterop"
        val headersDir = "${projectDir}/multik_jni/src/main/headers/"
        val cppDir = "${projectDir}/multik_jni/src/main/cpp"
        headers("$headersDir/mk_math.h", "$headersDir/mk_linalg.h")
        defFile(project.file(("$cinteropDir/libmultik.def")))

        extraOpts("-Xsource-compiler-option", "-std=c++14")
        extraOpts("-Xsource-compiler-option", "-I$headersDir")
        extraOpts("-Xsource-compiler-option", "-I${buildDir}/cmake-build/openblas-install/include")
        extraOpts("-Xcompile-source", "$cppDir/mk_math.cpp")
        extraOpts("-Xcompile-source", "$cppDir/mk_linalg.cpp")
    }
}
