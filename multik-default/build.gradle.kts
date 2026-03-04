@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)

plugins {
    id("multik.all-targets-kmp")
    id("multik.host-native-kmp")
    id("multik.publishing")
}

kotlin {
    applyDefaultHierarchyTemplate()

    sourceSets {
        commonMain {
            dependencies {
                api(project(":multik-core"))
                implementation(project(":multik-kotlin"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        jvmMain {
            dependencies {
                api(project(":multik-openblas"))
            }
        }

        val desktopMain by creating {
            dependsOn(commonMain.get())
            dependencies { api(project(":multik-openblas")) }
        }
        val desktopTest by creating { dependsOn(commonTest.get()) }

        getByName("macosArm64Main").dependsOn(desktopMain)
        getByName("macosX64Main").dependsOn(desktopMain)
        getByName("linuxX64Main").dependsOn(desktopMain)
        getByName("mingwX64Main").dependsOn(desktopMain)
        getByName("macosArm64Test").dependsOn(desktopTest)
        getByName("macosX64Test").dependsOn(desktopTest)
        getByName("linuxX64Test").dependsOn(desktopTest)
        getByName("mingwX64Test").dependsOn(desktopTest)
    }
}
