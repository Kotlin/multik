/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

plugins {
    val dokka_version: String by System.getProperties()

    id("org.jetbrains.dokka") version dokka_version
}

kotlin {
    explicitApi()
}

val common_csv_version: String by project
dependencies {
    implementation(kotlin("reflect"))
    implementation("org.apache.commons:commons-csv:$common_csv_version")
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