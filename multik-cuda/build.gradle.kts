repositories {
    mavenCentral()
}

fun getOsString(): String {
    val vendor = System.getProperty("java.vendor")
    if ("The Android Project" == vendor) {
        return "android"
    } else {
        var osName = System.getProperty("os.name")
        osName = osName.toLowerCase()
        if (osName.startsWith("windows")) {
            return "windows"
        } else if (osName.startsWith("mac os")) {
            return "apple"
        } else if (osName.startsWith("linux")) {
            return "linux"
        } else if (osName.startsWith("sun")) {
            return "sun"
        }
        return "unknown"
    }
}

fun getArchString(): String {
    var osArch = System.getProperty("os.arch")
    osArch = osArch.toLowerCase()

    if ("i386" == osArch || "x86" == osArch || "i686" == osArch) {
        return "x86"
    } else if (osArch.startsWith("amd64") || osArch.startsWith("x86_64")) {
        return "x86_64"
    } else if (osArch.startsWith("arm64")) {
        return "arm64"
    } else if (osArch.startsWith("arm")) {
        return "arm"
    } else if ("ppc" == osArch || "powerpc" == osArch) {
        return "ppc"
    } else if (osArch.startsWith("ppc")) {
        return "ppc_64"
    } else if (osArch.startsWith("sparc")) {
        return "sparc"
    } else if (osArch.startsWith("mips64")) {
        return "mips64"
    } else if (osArch.startsWith("mips")) {
        return "mips"
    } else if (osArch.contains("risc")) {
        return "risc"
    }
    return "unknown"
}

dependencies {
    api(project(":multik-api"))

    val classifier = getOsString() + "-" + getArchString()
    val jCudaVersion = "11.2.0"

    implementation(group = "org.jcuda", name = "jcuda", version = jCudaVersion) {
        isTransitive = false
    }
    implementation(group = "org.jcuda", name = "jcublas", version = jCudaVersion) {
        isTransitive = false
    }

    implementation(group = "org.jcuda", name = "jcuda-natives",
            classifier = classifier, version = jCudaVersion)
    implementation(group = "org.jcuda", name = "jcublas-natives",
            classifier = classifier, version = jCudaVersion)
}