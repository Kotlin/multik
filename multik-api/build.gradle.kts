/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    kotlin("multiplatform")
    id("org.jetbrains.dokka") version "1.4.32"
    id("maven-publish")
}

repositories {
    mavenCentral()
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
    }
    val hostOs = System.getProperty("os.name")
    val isMingwX64 = hostOs.startsWith("Windows")
    val nativeTarget = when {
        hostOs == "Mac OS X" -> macosX64("native")
        hostOs == "Linux" -> linuxX64("native")
        isMingwX64 -> mingwX64("native")
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }
    iosArm64()
    iosX64()

    sourceSets {
        val commonMain by getting {
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test-common"))
                implementation(kotlin("test-annotations-common"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("reflect"))
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit"))
            }
        }
        val nativeCommonMain by creating {
            dependsOn(commonMain)
        }

        val nativeMain by getting {
            dependsOn(nativeCommonMain)
        }
        val nativeTest by getting
        val iosArm64Main by getting {
            dependsOn(nativeCommonMain)
        }
        val iosX64Main by getting {
            dependsOn(nativeCommonMain)
        }

    }
}



tasks.dokkaHtml.configure {
    outputDirectory.set(rootProject.buildDir.resolve("dokka"))

    dokkaSourceSets {
        configureEach {
            includeNonPublic.set(false)
            skipEmptyPackages.set(false)
            jdkVersion.set(8)
            noStdlibLink.set(false)
            noJdkLink.set(false)
            samples.from(files("src/test/kotlin/samples/creation.kt"))
        }
    }
}