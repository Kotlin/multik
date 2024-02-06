import java.time.Duration

buildscript {
    repositories {
        mavenCentral()
    }
}

plugins {
    val kotlin_version: String by System.getProperties()
    val nexus_version: String by System.getProperties()

    kotlin("multiplatform") version kotlin_version apply false
    id("io.github.gradle-nexus.publish-plugin") version nexus_version
}

val kotlin_version: String by System.getProperties()
val multik_version: String by project
val unpublished = listOf("multik")

allprojects {
    repositories {
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = multik_version
}

configure(subprojects.filter { it.name !in unpublished }) {
    apply("$rootDir/gradle/publish.gradle")
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.js.npm.tasks.KotlinNpmInstallTask>().configureEach {
    args.add("--ignore-engines")
}

val sonatypeUser: String = System.getenv("SONATYPE_USER") ?: ""
val sonatypePassword: String = System.getenv("SONATYPE_PASSWORD") ?: ""

nexusPublishing {
    packageGroup.set(project.group.toString())
    this.repositories {
        sonatype {
            username.set(sonatypeUser)
            password.set(sonatypePassword)
            repositoryDescription.set("kotlinx.multik staging repository, version: $version")
        }
    }

    transitionCheckOptions {
        maxRetries.set(100)
        delayBetween.set(Duration.ofSeconds(5))
    }
}