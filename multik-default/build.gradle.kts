plugins {
    kotlin("multiplatform")
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
//        withJava()
//        val jvmJar by tasks.getting(org.gradle.jvm.tasks.Jar::class) {
//            doFirst {
//                manifest {
//                    attributes["Implementation-Title"] = project.name
//                    attributes["Implementation-Version"] = project.version
//                }
//                from(configurations.getByName("runtimeClasspath").map { if (it.isDirectory) it else zipTree(it) })
//            }
//        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":multik-api"))
                implementation(project(":multik-jvm"))
                implementation(project(":multik-native"))
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("reflect"))
            }
        }
    }
}

