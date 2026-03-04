import com.vanniktech.maven.publish.JavadocJar
import com.vanniktech.maven.publish.KotlinMultiplatform

plugins {
    id("com.vanniktech.maven.publish")
    id("multik.dokka")
    signing
}

mavenPublishing {
    publishToMavenCentral(true)

    configure(KotlinMultiplatform(javadocJar = JavadocJar.Dokka("dokkaGeneratePublicationHtml")))

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
                id.set("devcrocod")
                name.set("Pavel Gorgulov")
                organization.set("JetBrains")
                organizationUrl.set("https://www.jetbrains.com")
            }
        }
        scm {
            url = "https://github.com/Kotlin/multik"
            connection = "scm:git:https://github.com/Kotlin/multik.git"
            developerConnection = "scm:git:git@github.com:Kotlin/multik.git"
        }
    }
}

afterEvaluate {
    signing {
        val signingKeyId: String? by project
        val signingKey: String? by project
        val signingPassword: String? by project
        useInMemoryPgpKeys(signingKeyId, signingKey, signingPassword)
        mavenPublishing.signAllPublications()
        isRequired = !signingKey.isNullOrBlank()
    }
}
