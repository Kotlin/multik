import org.jetbrains.kotlin.gradle.dsl.ExplicitApiMode
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    kotlin("multiplatform")
}

kotlin {
    explicitApi = ExplicitApiMode.Strict

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm {
        compilerOptions.jvmTarget = JvmTarget.JVM_1_8

        tasks.withType<Test>().configureEach {
            useJUnitPlatform()
        }
    }

    targets.withType<KotlinNativeTarget> {
        binaries.getTest("debug").apply {
            debuggable = false
            optimized = true
        }
    }
}