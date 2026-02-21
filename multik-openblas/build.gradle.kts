import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.konan.target.Family.LINUX

plugins {
    id("multik.host-native-kmp")
    id("multik.cmake-build")
    id("multik.publishing")
}

kotlin {
    jvm {
        val jvmTest by tasks.getting(Test::class) {
            systemProperty("java.library.path", "$projectDir/build/cmake-build")
        }
        val jvmJar by tasks.getting(Jar::class) {
            dependsOn("copyNativeLibs")
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

    @Suppress("UNCHECKED_CAST")
    val hostTarget = project.extra["hostNativeTarget"] as KotlinNativeTarget

    hostTarget.apply {
        compilations.getByName("main") {
            cinterops {
                val libmultik by creating {
                    val cinteropDir = "${projectDir}/cinterop"
                    val headersDir = "${projectDir}/multik_jni/src/main/headers"
                    val cppDir = "${projectDir}/multik_jni/src/main/cpp"
                    val openblasIncludeDir = "${projectDir}/build/cmake-build/openblas-install/include"

                    definitionFile.set(project.file("$cinteropDir/libmultik.def"))
                    headers("$headersDir/mk_math.h", "$headersDir/mk_linalg.h", "$headersDir/mk_stat.h")

                    // Include directories for cinterop header analysis
                    includeDirs.allHeaders(headersDir, openblasIncludeDir)

                    // Compiler options for cinterop header analysis
                    if (konanTarget.family == LINUX) {
                        compilerOpts("-DFORCE_OPENBLAS_COMPLEX_STRUCT=1")
                    }

                    // Compiler options for C++ source files compiled into the binding
                    extraOpts("-Xsource-compiler-option", "-std=c++14")
                    extraOpts("-Xsource-compiler-option", "-I$headersDir")
                    extraOpts("-Xsource-compiler-option", "-I$openblasIncludeDir")
                    if (konanTarget.family == LINUX) {
                        extraOpts("-Xsource-compiler-option", "-DFORCE_OPENBLAS_COMPLEX_STRUCT=1")
                    }

                    // C++ source files to compile into the cinterop binding
                    extraOpts("-Xcompile-source", "$cppDir/mk_math.cpp")
                    extraOpts("-Xcompile-source", "$cppDir/mk_linalg.cpp")
                    extraOpts("-Xcompile-source", "$cppDir/mk_stat.cpp")
                }
            }
        }
    }

    sourceSets {
        commonMain {
            dependencies {
                api(project(":multik-core"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
    }
}

tasks.withType(CInteropProcess::class.java).configureEach { dependsOn("build_cmake") }
