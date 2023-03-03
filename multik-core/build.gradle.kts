/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
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
    wasm {
        browser {
            testTask {
                /*
                    https://youtrack.jetbrains.com/issue/KT-56633
                    https://youtrack.jetbrains.com/issue/KT-56159
                 */
                this.enabled = false // fixed in 1.9.0/1.9.20
            }
        }
    }
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