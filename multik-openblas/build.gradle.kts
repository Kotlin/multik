@file:OptIn(ExperimentalKotlinGradlePluginApi::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.konan.target.Family.LINUX

plugins {
    kotlin("multiplatform")
}

apply(from = "$rootDir/gradle/multik_jni-cmake.gradle")

kotlin {
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        testRuns["test"].executionTask.configure {
            useJUnit()
        }
        val jvmTest by tasks.getting(Test::class) {
            systemProperty("java.library.path", "$projectDir/build/cmake-build")
        }
        val jvmJar by tasks.getting(Jar::class) {
            from("$projectDir/build/libs") {
                include("libmultik_jni-androidArm64.so")
                into("lib/arm64-v8a")
            }
            from("$projectDir/build/libs") {
                include("libmultik_jni-linuxX64.so")
                into("lib/linuxX64")
            }
            from("$projectDir/build/libs") {
                include("libmultik_jni-macosArm64.dylib")
                into("lib/macosArm64")
            }
            from("$projectDir/build/libs") {
                include("libmultik_jni-macosX64.dylib")
                into("lib/macosX64")
            }
            from("$projectDir/build/libs") {
                include("libmultik_jni-mingwX64.dll")
                into("lib/mingwX64")
            }

        }
    }
    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    val hostTarget = when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64 {
            binaries {
                framework { baseName = "multik-openblas" }
            }
        }
        hostOs == "Mac OS X" && hostArch == "aarch64" -> macosArm64 {
            binaries {
                framework { baseName = "multik-openblas" }
            }
        }
        hostOs == "Linux" -> linuxX64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }

    hostTarget.apply {
        compilations.getByName("main") {
            cinterops {
                val libmultik by creating {
                    val cinteropDir = "${projectDir}/cinterop"
                    val headersDir = "${projectDir}/multik_jni/src/main/headers/"
                    val cppDir = "${projectDir}/multik_jni/src/main/cpp"
                    headers("$headersDir/mk_math.h", "$headersDir/mk_linalg.h", "$headersDir/mk_stat.h")
                    defFile(project.file(("$cinteropDir/libmultik.def")))

                    when (konanTarget.family) {
                        LINUX -> extraOpts("-Xsource-compiler-option", "-DFORCE_OPENBLAS_COMPLEX_STRUCT=1")
                        else -> {
                            // Nothing
                        }
                    }

                    extraOpts("-Xsource-compiler-option", "-std=c++14")
                    extraOpts("-Xsource-compiler-option", "-I$headersDir")
                    extraOpts("-Xsource-compiler-option", "-I${projectDir}/build/cmake-build/openblas-install/include")
                    extraOpts("-Xcompile-source", "$cppDir/mk_math.cpp")
                    extraOpts("-Xcompile-source", "$cppDir/mk_linalg.cpp")
                    extraOpts("-Xcompile-source", "$cppDir/mk_stat.cpp")
                }
            }
        }
        binaries.all {
            freeCompilerArgs = freeCompilerArgs + "-Xallocator=std"
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
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting
    }
}

tasks.withType(CInteropProcess::class.java).configureEach { dependsOn("build_cmake") }
