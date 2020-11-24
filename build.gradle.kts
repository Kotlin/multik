import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

buildscript {
    repositories {
        jcenter()
    }
}

plugins {
    val kotlinVersion: String by System.getProperties()
    kotlin("jvm") version kotlinVersion

    id("org.jetbrains.dokka") version "1.4.10.2"

    id("com.jfrog.bintray") version "1.8.5"
}

val kotlinVersion: String by System.getProperties()
val multikVersion by extra("0.0.1")
val unpublished = listOf("examples", "benchmarks")

allprojects {
    apply(plugin = "maven-publish")
    repositories {
        jcenter()
    }

    group = "org.jetbrains.multik"
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

configure(subprojects.filter { it.name !in unpublished }) {
    apply("$rootDir/gradle/dokka.gradle")
    apply("$rootDir/gradle/publish.gradle")
}