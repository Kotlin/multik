dependencies {
    api(project(":multik-api"))
    implementation(project(":multik-jvm"))
    implementation(project(":multik-native"))
}

val fatJar = task("fatJar", type = Jar::class) {
    manifest {
        attributes(
            mapOf("Implementation-Title" to project.name,
                "Implementation-Version" to project.version)
        )
    }
    manifest {
        attributes["Implementation-Title"] = "Default implementation"
        attributes["Implementation-Version"] = archiveVersion
    }
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
    with(tasks.jar.get() as CopySpec)
}

tasks.named("build") { dependsOn(fatJar) }
