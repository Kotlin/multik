pluginManagement {
    repositories {
        gradlePluginPortal()
    }
}

rootProject.name = "multik"
include(
    ":multik-core",
    ":multik-default",
    ":multik-kotlin",
    ":multik-native",
)
