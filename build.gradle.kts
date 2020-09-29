import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

buildscript {
    repositories {
        jcenter()
    }
}

plugins {
    val kotlinVersion: String by System.getProperties()
    kotlin("jvm") version kotlinVersion

    id("org.jetbrains.dokka") version "1.4.0"

    `maven-publish`
}

val kotlinVersion: String by System.getProperties()
val multikVersion by extra("0.0.1")

allprojects {

    repositories {
        jcenter()
    }

    group = "multik"
    version = multikVersion

    tasks.withType<KotlinCompile> {
        kotlinOptions.jvmTarget = "1.8"
    }

//    TODO(dokka tasks)
}

subprojects {
    apply(plugin = "kotlin")

    dependencies {
        testImplementation(kotlin("test"))
        testImplementation(kotlin("test-junit"))
    }
}
