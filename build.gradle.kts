val multikVersion: String by project

allprojects {
    group = "org.jetbrains.kotlinx"
    version = multikVersion
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.js.npm.tasks.KotlinNpmInstallTask>().configureEach {
    args.add("--ignore-engines")
}