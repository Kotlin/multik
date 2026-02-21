import org.jetbrains.dokka.gradle.engine.parameters.VisibilityModifier

plugins {
    id("org.jetbrains.dokka")
}

dokka {
    moduleName.set("Multik")

    dokkaSourceSets.configureEach {
        pluginsConfiguration.html {
            footerMessage = "Copyright © 2020-2026 JetBrains s.r.o."
        }

        sourceLink {
            localDirectory = projectDir.resolve("src")
            remoteUrl("https://github.com/Kotlin/multik")
            remoteLineSuffix.set("#L")
            skipDeprecated = false
            jdkVersion = 8
        }

        documentedVisibilities(VisibilityModifier.Public)
    }
}