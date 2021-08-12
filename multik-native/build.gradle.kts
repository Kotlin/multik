/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

apply(from = "$rootDir/gradle/openblas.gradle")

dependencies {
    api(project(":multik-api"))
}

tasks.jar {
    dependsOn("multik_jni:build")
    doFirst {
        from(fileTree("${project.childProjects["multik_jni"]!!.buildDir}/lib").files)
        exclude("*.jar")
    }
}

tasks.test {
    dependsOn("multik_jni:build")

    doFirst{
        copy {
            from(fileTree("${project.childProjects["multik_jni"]!!.buildDir}/lib").files)
            into("$buildDir/libs")
        }
    }

    systemProperty("java.library.path", "$buildDir/libs")
}
