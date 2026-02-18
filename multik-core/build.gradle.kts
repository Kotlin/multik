@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)

import org.jetbrains.dokka.gradle.engine.parameters.VisibilityModifier
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("multiplatform")
    alias(libs.plugins.dokka)
    alias(libs.plugins.korro)
}

repositories {
    mavenCentral()
}

kotlin {
    explicitApi()

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm {
        compilerOptions.jvmTarget = JvmTarget.JVM_1_8
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
        browser {
            testTask {
                enabled = false
            }
        }
        nodejs {
            testTask {
                enabled = false
            }
        }
        binaries.library()
    }
    js(IR) {
        browser {
            testTask {
                useMocha()
            }
        }
        nodejs {
            testTask {
                useMocha()
            }
        }
        binaries.library()
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
                api(kotlin("reflect"))
                api(libs.commons.csv)
                api(libs.bio.npy)
            }
        }
    }
}

korro {
    docs = fileTree(rootProject.rootDir) {
        include("docs/topics/*.md")
        include("docs/topics/gettingStarted/*md")
        include("docs/topics/userGuide/*md")
        include("docs/topics/apiDocs/*md")
    }

    samples = fileTree(project.projectDir) {
        include("src/commonTest/kotlin/samples/docs/*.kt")
        include("src/commonTest/kotlin/samples/docs/userGuide/*.kt")
        include("src/commonTest/kotlin/samples/docs/apiDocs/*.kt")
    }
}

dokka {
    moduleName.set("Multik")

    dokkaSourceSets.configureEach {
        sourceLink {
            localDirectory.set(file("src/main/kotlin"))
            remoteUrl("https://github.com/Kotlin/multik")
            remoteLineSuffix.set("#L")
            documentedVisibilities(VisibilityModifier.Public, VisibilityModifier.Protected)
            skipDeprecated = false
            jdkVersion = 8
            samples.from(files("src/commonTest/kotlin/samples/creation.kt"))
        }
    }
    dokkaPublications.html {
        outputDirectory = project.rootProject.layout.buildDirectory.dir("dokka")
        suppressObviousFunctions = true
        suppressInheritedMembers = true
    }
}
