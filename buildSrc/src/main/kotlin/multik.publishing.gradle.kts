plugins {
    `maven-publish`
}

val javadocJar by tasks.registering(Jar::class) {
    archiveClassifier.set("javadoc")
}

fun MavenPublication.configurePom() {
    pom {
        name.set(project.name)
        description.set("Multidimensional array library for Kotlin.")
        url.set("https://github.com/Kotlin/multik")

        licenses {
            license {
                name.set("The Apache Software License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("JetBrains")
                name.set("JetBrains Team")
                organization.set("JetBrains")
                organizationUrl.set("https://www.jetbrains.com")
            }
        }
        scm {
            url.set("https://github.com/Kotlin/multik")
        }
    }
}

/**
 * Merges the platform publication's POM into the kotlinMultiplatform publication,
 * so that consuming Gradle projects get correct metadata.
 *
 * Adapted from https://github.com/Kotlin/kotlinx.serialization/blob/master/gradle/publish-mpp-root-module-in-platform.gradle
 */
fun publishPlatformArtifactsInRootModule(platformPublication: MavenPublication) {
    afterEvaluate {
        var platformXml: org.gradle.api.XmlProvider? = null
        platformPublication.pom.withXml { platformXml = this }

        publishing.publications.named<MavenPublication>("kotlinMultiplatform") {
            pom.withXml {
                val root = asNode()
                root.children().toList().forEach { root.remove(it as groovy.util.Node) }
                platformXml!!.asNode().children().forEach { root.append(it as groovy.util.Node) }

                ((root.get("artifactId") as groovy.util.NodeList)[0] as groovy.util.Node)
                    .setValue(this@named.artifactId)

                root.appendNode("packaging", "pom")

                val dependencies = (root.get("dependencies") as groovy.util.NodeList)[0] as groovy.util.Node
                dependencies.children().toList().forEach { dependencies.remove(it as groovy.util.Node) }
                val dep = dependencies.appendNode("dependency")
                dep.appendNode("groupId", platformPublication.groupId)
                dep.appendNode("artifactId", platformPublication.artifactId)
                dep.appendNode("version", platformPublication.version)
                dep.appendNode("scope", "compile")
            }
        }

        tasks.matching { it.name == "generatePomFileForKotlinMultiplatformPublication" }.configureEach {
            dependsOn("generatePomFileFor${platformPublication.name.replaceFirstChar { it.uppercase() }}Publication")
        }
    }
}

publishing {
    val variantName = project.name

    publications.withType<MavenPublication>().configureEach {
        when (name) {
            "kotlinMultiplatform" -> {
                artifactId = variantName
            }
            "metadata", "jvm", "js" -> {
                artifactId = "$variantName-$name"
            }
        }
        configurePom()
        groupId = project.group.toString()
        version = project.version.toString()

        if (name != "kotlinMultiplatform") {
            artifact(javadocJar)
        }
    }

    afterEvaluate {
        publications.findByName("jvm")?.let { jvmPub ->
            publishPlatformArtifactsInRootModule(jvmPub as MavenPublication)
        }
    }

    repositories {
        maven {
            url = uri("https://oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = System.getenv("SONATYPE_USER") ?: ""
                password = System.getenv("SONATYPE_PASSWORD") ?: ""
            }
        }
    }
}

val signingKeyId = System.getenv("SIGN_KEY_ID")
if (signingKeyId != null) {
    apply(plugin = "signing")

    configure<org.gradle.plugins.signing.SigningExtension> {
        useInMemoryPgpKeys(
            signingKeyId,
            System.getenv("SIGN_KEY_PRIVATE"),
            System.getenv("SIGN_KEY_PASSPHRASE")
        )
        sign(publishing.publications)
    }
}