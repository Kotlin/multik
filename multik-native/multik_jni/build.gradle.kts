/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

import org.gradle.internal.jvm.Jvm

plugins {
    `cpp-library`
}

apply(from = "$rootDir/gradle/openblas.gradle")

val gccPath: String? = System.getenv("MinGW_x64_Bin_Path")
val gfortranAndQuadmathPath: String? = System.getenv("path_to_gfortran")

val linkList: List<String> = mutableListOf("$buildDir/openblas/lib/libopenblas.a").apply {
    if (gfortranAndQuadmathPath != null) {
        this.add("$gfortranAndQuadmathPath/libgfortran.a")
        this.add("$gfortranAndQuadmathPath/libquadmath.a")
    }
}

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
            linkList +
                    if (it.operatingSystem.isWindows)
                        listOf("-static-libgcc", "-static-libstdc++", "-static", "-lpthread")
                    else
                        emptyList()
        }
    )
}