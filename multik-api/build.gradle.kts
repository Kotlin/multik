plugins {
    id("org.jetbrains.dokka") version "1.4.10.2"
}

kotlin {
    explicitApi()
}

dependencies {
    implementation(kotlin("reflect"))
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