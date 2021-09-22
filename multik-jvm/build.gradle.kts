plugins {
    kotlin("multiplatform")
    id("maven-publish")
}


kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        testRuns["test"].executionTask.configure {
            useJUnit()
        }
        withJava()
    }
    val hostOs = System.getProperty("os.name")
    val isMingwX64 = hostOs.startsWith("Windows")
    val nativeTarget = when {
        hostOs == "Mac OS X" -> macosX64("native")
        hostOs == "Linux" -> linuxX64("native")
        isMingwX64 -> mingwX64("native")
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }

    nativeTarget.apply {
        binaries {
            framework {
                baseName = "library"
            }
        }
    }
    iosArm64 {
        binaries {
            framework {
                baseName = "library"
            }
        }
    }
    iosX64 {
        binaries {
            framework {
                baseName = "library"
            }
        }
    }

    js {
        browser()
        nodejs()
    }

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget> {
        binaries.all {
            freeCompilerArgs = freeCompilerArgs + "-Xallocator=mimalloc"
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":multik-api"))
            }
        }
        val commonTest by getting {
            dependencies {
                api(project(":multik-api"))
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
                implementation(kotlin("reflect"))
                implementation(kotlin("test-junit"))
            }
        }
        val nativeMain by getting
        val nativeTest by getting
        val iosArm64Main by getting
        val iosArm64Test by getting
        val iosX64Main by getting
        val iosX64Test by getting
    }
}

