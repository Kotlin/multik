buildscript {
    repositories {
        mavenCentral()
    }
}

plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
}

val multikVersion: String by project
val unpublished = listOf("multik")

allprojects {
    repositories {
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = multikVersion
}

configure(subprojects.filter { it.name !in unpublished }) {
    apply("$rootDir/gradle/publish.gradle")
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.js.npm.tasks.KotlinNpmInstallTask>().configureEach {
    args.add("--ignore-engines")
}

//val sonatypeUser: String = System.getenv("SONATYPE_USER") ?: ""
//val sonatypePassword: String = System.getenv("SONATYPE_PASSWORD") ?: ""

//nexusPublishing {
//    packageGroup.set(project.group.toString())
//    this.repositories {
//        sonatype {
//            username.set(sonatypeUser)
//            password.set(sonatypePassword)
//            repositoryDescription.set("kotlinx.multik staging repository, version: $version")
//        }
//    }
//
//    transitionCheckOptions {
//        maxRetries.set(100)
//        delayBetween.set(Duration.ofSeconds(5))
//    }
//}