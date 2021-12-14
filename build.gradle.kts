/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

buildscript {
    repositories {
        mavenCentral()
    }
}

plugins {
    val kotlin_version: String by System.getProperties()
    val nexus_version: String by System.getProperties()

    kotlin("multiplatform") version kotlin_version apply false
    id("io.codearte.nexus-staging") version nexus_version
}

val kotlin_version: String by System.getProperties()
val multik_version: String by project
val unpublished = listOf("multik", "multik_jni")

allprojects {
    repositories {
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = multik_version

}

configure(subprojects.filter { it.name !in unpublished }) {
    //apply("$rootDir/gradle/publish.gradle")
}