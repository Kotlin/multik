import org.gradle.internal.jvm.Jvm

plugins {
    `cpp-library`
}

apply(from = "$rootDir/gradle/openblas.gradle")

val gccPath: String? = System.getenv("MinGW_x64_Bin_Path")

library {
    source.from(file("src/main/cpp"))

    toolChains.configureEach {
        if (this is Gcc && gccPath != null) {
            this.path(gccPath)
        }
    }

    targetMachines.set(
        listOf(
            machines.linux.x86_64,
            machines.windows.x86_64,
            machines.macOS.x86_64
        )
    )

    linkage.set(listOf(Linkage.SHARED))

    binaries.configureEach {
        compileTask.get().compilerArgs.addAll(compileTask.get().targetPlatform.map {
            listOf(
                "-std=c++14", "-O3", "-fno-exceptions", "-ffast-math",
                "-I", "${Jvm.current().javaHome.canonicalPath}/include",
                "-I", "$buildDir/openblas/include/"
            ) + when {
                it.operatingSystem.isMacOsX -> listOf("-I", "${Jvm.current().javaHome.canonicalPath}/include/darwin")
                it.operatingSystem.isLinux -> listOf("-I", "${Jvm.current().javaHome.canonicalPath}/include/linux")
                it.operatingSystem.isWindows -> listOf(
                    "-static-libgcc", "-static-libstdc++", "-static", "-lpthread",
                    "-I", "${Jvm.current().javaHome.canonicalPath}/include/win32"
                )
                else -> emptyList()
            }
        })
    }


}


tasks.withType(CppCompile::class.java).configureEach { dependsOn("installOpenBlas") }

tasks.withType(LinkSharedLibrary::class.java).configureEach { linkerArgs.addAll(listOf("$buildDir/openblas/lib/libopenblas.a")) }