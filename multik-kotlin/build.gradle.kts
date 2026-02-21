plugins {
    id("multik.all-targets-kmp")
    id("multik.publishing")
}

kotlin {
    sourceSets {
        commonMain {
            dependencies {
                api(project(":multik-core"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
    }
}