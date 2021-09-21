/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

buildscript {
    repositories {
        mavenCentral()
    }
}

plugins {
    val kotlinVersion: String by System.getProperties()
    kotlin("jvm") version kotlinVersion

    id("io.codearte.nexus-staging") version "0.22.0"
}

val kotlinVersion: String by System.getProperties()
val multikVersion: String by project
val unpublished = listOf("multik", "multik_jni")

allprojects {
    repositories {
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = multikVersion

    tasks.withType<KotlinCompile> {
        kotlinOptions.jvmTarget = "1.8"
    }
}

subprojects {
    if (!this.name.contains("jni")) {
        apply(plugin = "kotlin")

        dependencies {
            testImplementation(kotlin("test"))
            testImplementation(kotlin("test-junit"))
        }
    }
}

configure(subprojects.filter { it.name !in unpublished }) {
    apply("$rootDir/gradle/publish.gradle")
}