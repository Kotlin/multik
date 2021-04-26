pluginManagement {
    repositories {
        gradlePluginPortal()
        jcenter()
    }
}

rootProject.name = "multik"
include(
    ":multik-api",
    ":multik-default",
    ":multik-jvm",
    ":multik-native",
    ":examples"
)
