/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    kotlin("multiplatform")
    val dokka_version: String by System.getProperties()

    id("org.jetbrains.dokka") version dokka_version
}

repositories {
    mavenCentral()
}

val common_csv_version: String by project

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
    mingwX64()
    linuxX64()
    macosX64()
    macosArm64()
    iosArm64()
    iosSimulatorArm64()
    iosX64()
    js(IR) {
        val timeoutMs = "1000000"
        browser{
            testTask {
                useMocha {
                    timeout = timeoutMs
                }
            }
        }
        nodejs{
            testTask {
                useMocha {
                    timeout = timeoutMs
                }
            }
        }
    }

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        binaries.all {
            freeCompilerArgs = freeCompilerArgs + "-Xallocator=mimalloc"
        }
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
            }
        }
        val nativeMain by creating {
            dependsOn(commonMain)
        }
        names.forEach { n ->
            if (n.contains("X64Main") || n.contains("Arm64Main")){
                this@sourceSets.getByName(n).apply{
                    dependsOn(nativeMain)
                }
            }
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