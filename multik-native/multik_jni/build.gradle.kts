/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

import org.gradle.internal.jvm.Jvm

plugins {
    `cpp-library`
}

apply(from = "$rootDir/gradle/openblas.gradle")

val gccMinGWPath: String? = System.getenv("MinGW_x64_Bin_Path")
val gccLibPath: String? = System.getenv("path_to_libgcc")
val gccDarwin: String? = System.getenv("path_to_gcc_darwin") ?: gccLibPath

library {
    source.from(file("src/main/cpp"))

    toolChains.configureEach {
        if (this is Gcc && gccMinGWPath != null) {
            this.path(gccMinGWPath)
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
                "-std=c++14", "-O3", "-fno-exceptions", "-ffast-math", "-fPIC",
                "-I", "${Jvm.current().javaHome.canonicalPath}/include",
                "-I", "$buildDir/openblas/include/"
            ) + when {
                it.operatingSystem.isMacOsX -> listOf("-I", "${Jvm.current().javaHome.canonicalPath}/include/darwin")
                it.operatingSystem.isLinux -> listOf("-I", "${Jvm.current().javaHome.canonicalPath}/include/linux")
                it.operatingSystem.isWindows -> listOf("-I", "${Jvm.current().javaHome.canonicalPath}/include/win32")
                else -> emptyList()
            }
        })
    }
}


tasks.withType(CppCompile::class.java).configureEach { dependsOn("installOpenBlas") }

tasks.withType(LinkSharedLibrary::class.java).configureEach {
    linkerArgs.addAll(
        targetPlatform.map {
            listOf("$buildDir/openblas/lib/libopenblas.a") +
                when {
                    it.operatingSystem.isWindows ->
                        listOf(
                            "-static-libgcc", "-static-libstdc++", "-static", "-lpthread",
                            "$gccLibPath/libgfortran.a", "$gccLibPath/libquadmath.a"
                        )
                    it.operatingSystem.isMacOsX ->
                        listOf(
                            "$gccDarwin/libgcc.a",
                            "$gccLibPath/libgfortran.a", "$gccLibPath/libquadmath.a"
                        )
                    it.operatingSystem.isLinux ->
                        listOf("$gccLibPath/libgfortran.a", "$gccLibPath/libquadmath.a")
                    else -> emptyList()
                }
        }
    )
}