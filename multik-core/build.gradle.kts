@file:OptIn(ExperimentalWasmDsl::class, ExperimentalKotlinGradlePluginApi::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.targets.js.dsl.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.targets.js.nodejs.NodeJsRootExtension

plugins {
    kotlin("multiplatform")
    val dokka_version: String by System.getProperties()
    val korro_version: String by System.getProperties()

    id("org.jetbrains.dokka") version dokka_version
    id("io.github.devcrocod.korro") version korro_version
}

repositories {
    mavenCentral()
}

val common_csv_version: String by project
val nodeJsVersion: String by project
val nodeDownloadUrl: String by project

kotlin {
    explicitApi()

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
    }
    mingwX64()
    linuxX64()
    macosX64()
    macosArm64()
    iosArm64()
    iosSimulatorArm64()
    iosX64()
    wasmJs {
        browser()
        nodejs()
        d8()
    }
    js(IR) {
        browser()
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("reflect"))
                implementation("org.apache.commons:commons-csv:$common_csv_version")
                implementation("org.jetbrains.bio:npy:0.3.5")
            }
        }
    }
}

rootProject.the<NodeJsRootExtension>().apply {
    nodeVersion = nodeJsVersion
    nodeDownloadBaseUrl = "https://nodejs.org/download/v8-canary"
}

korro {
    docs = fileTree(rootProject.rootDir) {
        include("docs/topics/*.md")
        include("docs/topics/gettingStarted/*md")
    }

    samples = fileTree(project.projectDir) {
        include("src/commonTest/kotlin/samples/docs/*.kt")
    }
}

tasks.dokkaHtml.configure {
    outputDirectory.set(rootProject.buildDir.resolve("dokka"))
    suppressObviousFunctions.set(true)
    suppressInheritedMembers.set(true)

    dokkaSourceSets {
        configureEach {
            documentedVisibilities.set(
                setOf(
                    org.jetbrains.dokka.DokkaConfiguration.Visibility.PUBLIC,
                    org.jetbrains.dokka.DokkaConfiguration.Visibility.PROTECTED
                )
            )
            skipDeprecated.set(false)
            jdkVersion.set(8)
            noStdlibLink.set(false)
            noJdkLink.set(false)
            samples.from(files("src/commonTest/kotlin/samples/creation.kt"))
        }
    }
}