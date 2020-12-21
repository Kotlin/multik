/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("me.champeau.gradle.jmh") version "0.5.0"
}

val compileJmhKotlin: KotlinCompile by tasks
compileJmhKotlin.kotlinOptions.jvmTarget = "1.8"

dependencies {
    implementation(project(":multik-api"))
    implementation(project(":multik-jvm"))
    implementation(project(":multik-native"))
    jmhImplementation(kotlin("stdlib"))
    jmhImplementation("pl.project13.scala:sbt-jmh-extras:0.3.3")
}

jmh {
    jmhVersion = "1.21"
//    duplicateClassesStrategy = DuplicatesStrategy.EXCLUDE
}