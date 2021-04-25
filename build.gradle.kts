/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

buildscript {
    repositories {
        mavenCentral()
    }
}

plugins {
    val kotlinVersion: String by System.getProperties()
    kotlin("multiplatform") version kotlinVersion apply false
    id("io.codearte.nexus-staging") version "0.22.0"
}

val kotlinVersion: String by System.getProperties()
val multikVersion: String by project
val unpublished = listOf("multik", "examples", "benchmarks")

allprojects {
    repositories {
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = multikVersion

}


configure(subprojects.filter { it.name !in unpublished }) {
    //apply("$rootDir/gradle/publish.gradle")
}