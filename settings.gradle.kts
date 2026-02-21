rootProject.name = "multik"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }

    plugins {
        id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

include(
    ":multik-core",
    ":multik-default",
    ":multik-kotlin",
    ":multik-openblas",
)
