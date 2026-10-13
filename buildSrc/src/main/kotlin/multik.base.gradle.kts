@file:OptIn(ExperimentalKotlinGradlePluginApi::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.ExplicitApiMode
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    kotlin("multiplatform")
    id("com.android.kotlin.multiplatform.library")
}

kotlin {
    explicitApi = ExplicitApiMode.Strict

    jvmToolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    applyDefaultHierarchyTemplate {
        common {
            group("jvmAndAndroid") {
                withJvm()
                withCompilations { it.platformType == KotlinPlatformType.androidJvm }
            }
        }
    }

    jvm {
        compilerOptions.jvmTarget = JvmTarget.JVM_11

        tasks.withType<Test>().configureEach {
            useJUnitPlatform()
        }
    }

    android {
        // A single segment per module: "default" is a Java keyword, so ".multik.default" is not a valid package.
        namespace = "org.jetbrains.kotlinx.${project.name.replace('-', '_')}"
        compileSdk = 37
        minSdk = 28
        aarMetadata.minCompileSdk = 28
        compilerOptions.jvmTarget = JvmTarget.JVM_11
    }

    targets.withType<KotlinNativeTarget> {
        binaries.getTest("debug").apply {
            debuggable = false
            optimized = true
        }
    }
}
