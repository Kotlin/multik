plugins {
    `kotlin-dsl`
}

repositories {
    mavenCentral()
    google()
    gradlePluginPortal()
}

dependencies {
    implementation(libs.kotlin.gradle)
    implementation(libs.android.gradle)
    implementation(libs.dokka.gradle)
    implementation(libs.maven.publish.gradle)
}