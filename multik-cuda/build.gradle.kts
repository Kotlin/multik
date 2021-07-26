repositories {
    mavenCentral()
}

val jcudaNativeLibs by configurations.creating
val jcudaLibs by configurations.creating

configurations {
    implementation.get().extendsFrom(jcudaLibs, jcudaNativeLibs)
}

dependencies {
    api(project(":multik-api"))

    implementation("io.github.microutils:kotlin-logging:+")
    implementation("org.slf4j:slf4j-simple:+")

    val jCudaVersion = "11.2.0"

    jcudaLibs(group = "org.jcuda", name = "jcuda", version = jCudaVersion) {
        isTransitive = false
    }
    jcudaLibs(group = "org.jcuda", name = "jcublas", version = jCudaVersion) {
        isTransitive = false
    }

    val archX86 = "x86_64"

    val classifierLinux = "linux-$archX86"
    val classifierWindows = "windows-$archX86"

    jcudaNativeLibs(group = "org.jcuda", name = "jcuda-natives", classifier = classifierLinux, version = jCudaVersion)
    jcudaNativeLibs(group = "org.jcuda", name = "jcublas-natives", classifier = classifierLinux, version = jCudaVersion)

    jcudaNativeLibs(group = "org.jcuda", name = "jcuda-natives", classifier = classifierWindows, version = jCudaVersion)
    jcudaNativeLibs(group = "org.jcuda", name = "jcublas-natives", classifier = classifierWindows, version = jCudaVersion)
}

tasks.register<Copy>("unzipNativeLibs") {
    from(jcudaNativeLibs.map {
        zipTree(it).matching {
            include("lib/")
        }
    })
    from(jcudaLibs.map {
        zipTree(it).matching {
            exclude("META-INF/")
        }
    })

    into("$buildDir/native_libs")
}

tasks.jar {
    dependsOn("unzipNativeLibs")
    from("$buildDir/native_libs")
}
