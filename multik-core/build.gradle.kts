plugins {
    id("multik.all-targets-kmp")
    id("multik.publishing")
    alias(libs.plugins.korro)
}

kotlin {
    sourceSets {
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        jvmMain {
            dependencies {
                api(libs.commons.csv)
                api(libs.bio.npy)
            }
        }
    }
}

korro {
    docs = fileTree(rootProject.rootDir) {
        include("README.md")
        include("docs/topics/*.md")
        include("docs/topics/gettingStarted/*md")
        include("docs/topics/userGuide/*md")
        include("docs/topics/apiDocs/*md")
    }

    samples = fileTree(project.projectDir) {
        include("src/commonTest/kotlin/samples/*.kt")
        include("src/commonTest/kotlin/samples/docs/*.kt")
        include("src/commonTest/kotlin/samples/docs/userGuide/*.kt")
        include("src/commonTest/kotlin/samples/docs/apiDocs/*.kt")
    }
}
