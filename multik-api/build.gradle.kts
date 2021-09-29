/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    id("org.jetbrains.dokka") version "1.5.0"
}

kotlin {
    explicitApi()
}

dependencies {
    implementation(kotlin("reflect"))
    implementation("org.apache.commons:commons-csv:1.9.0")
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