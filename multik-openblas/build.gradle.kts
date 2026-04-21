import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.konan.target.Family.LINUX

plugins {
    id("multik.host-native-kmp")
    id("multik.cmake-build")
    id("multik.publishing")
}

val hostTargetName = (project.extra["hostNativeTarget"] as KotlinNativeTarget).name

kotlin {
    jvm {
        val jvmTest by tasks.getting(Test::class) {
            systemProperty("java.library.path", "$projectDir/build/cmake-build")
        }
        val jvmJar by tasks.getting(Jar::class) {
            dependsOn("copyNativeLibs")
            from("${rootProject.projectDir}") {
                include("NOTICE")
                into("META-INF")
            }
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
                include("libmultik_jni-mingwX64.dll")
                into("lib/mingwX64")
            }
        }
    }

    targets.withType<KotlinNativeTarget>().configureEach {
        val isHostTarget = name == hostTargetName
        compilations.getByName("main") {
            cinterops.create("libmultik") {
                val cinteropDir = "${projectDir}/cinterop"
                val headersDir = "${projectDir}/multik_jni/src/main/headers"

                definitionFile.set(project.file("$cinteropDir/libmultik.def"))
                headers("$headersDir/mk_math.h", "$headersDir/mk_linalg.h", "$headersDir/mk_stat.h")
                includeDirs.allHeaders(headersDir)

                if (isHostTarget) {
                    val cppDir = "${projectDir}/multik_jni/src/main/cpp"
                    val openblasIncludeDir = "${projectDir}/build/cmake-build/openblas-install/include"

                    includeDirs.allHeaders(openblasIncludeDir)

                    if (konanTarget.family == LINUX) {
                        compilerOpts("-DFORCE_OPENBLAS_COMPLEX_STRUCT=1")
                    }

                    extraOpts("-Xccall-mode", "indirect")

                    extraOpts("-Xsource-compiler-option", "-std=c++14")
                    extraOpts("-Xsource-compiler-option", "-I$headersDir")
                    extraOpts("-Xsource-compiler-option", "-I$openblasIncludeDir")
                    if (konanTarget.family == LINUX) {
                        extraOpts("-Xsource-compiler-option", "-DFORCE_OPENBLAS_COMPLEX_STRUCT=1")
                    }

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

tasks.withType(CInteropProcess::class.java).configureEach {
    if (name.contains(hostTargetName, ignoreCase = true)) {
        dependsOn("build_cmake")
    }
}
