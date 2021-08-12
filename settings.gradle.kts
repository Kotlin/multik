pluginManagement {
    repositories {
        gradlePluginPortal()
    }
}

rootProject.name = "multik"
include(
    ":multik-api",
    ":multik-default",
    ":multik-jvm",
    ":multik-native",
    "multik-native:multik_jni",
    ":examples"
)
