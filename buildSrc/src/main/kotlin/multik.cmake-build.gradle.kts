import org.jetbrains.kotlinx.multik.builds.HomebrewGccDetection
import org.jetbrains.kotlinx.multik.builds.HostDetection

val cmakePath = "${rootDir}/multik-openblas/multik_jni"
val cmakeBuildDir = layout.buildDirectory.dir("cmake-build").map { it.asFile.absolutePath }

val cmakeCCompiler = System.getenv("CMAKE_C_COMPILER")
    ?: HomebrewGccDetection.cCompiler
    ?: "gcc"
val cmakeCxxCompiler = System.getenv("CMAKE_CXX_COMPILER")
    ?: HomebrewGccDetection.cxxCompiler
    ?: "g++"
val gccLibPath = System.getenv("GCC_LIB_Path")
    ?: HomebrewGccDetection.libPath
    ?: run {
        try {
            val process = ProcessBuilder(cmakeCCompiler, "-print-libgcc-file-name").start()
            val output = process.inputStream.bufferedReader().readText().trim()
            process.waitFor()
            if (output.isNotEmpty()) File(output).parent ?: "" else ""
        } catch (_: Exception) { "" }
    }
val targetOS = HostDetection.targetOSForCMake

val createBuildDir by tasks.registering {
    description = "Creates CMake build directories"
    doLast {
        layout.buildDirectory.get().asFile.mkdirs()
        layout.buildDirectory.dir("cmake-build").get().asFile.mkdirs()
    }
}

val configureCmake by tasks.registering(Exec::class) {
    description = "Configures CMake project for native JNI build"
    group = "build"
    dependsOn(createBuildDir)

    val args = mutableListOf(
        "cmake",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=$cmakeCCompiler",
        "-DCMAKE_CXX_COMPILER=$cmakeCxxCompiler",
        "-DGCC_LIB_PATH=$gccLibPath",
        "-DTARGET_OS=$targetOS",
        "-S", cmakePath,
        "-B", cmakeBuildDir.get()
    )
    if (HostDetection.isWindows) {
        args.addAll(listOf("-G", "CodeBlocks - MinGW Makefiles"))
    }
    commandLine(args)
}

val compileCmake by tasks.registering(Exec::class) {
    description = "Builds native JNI library with CMake"
    group = "build"
    dependsOn(configureCmake)

    val processors = Runtime.getRuntime().availableProcessors().toString()
    commandLine("cmake", "--build", cmakeBuildDir.get(), "--target", "multik_jni-$targetOS", "--", "-j", processors)
}

val copyNativeLibs by tasks.registering {
    description = "Copies compiled native libraries to the libs directory"
    group = "build"
    dependsOn(compileCmake)
    doLast {
        copy {
            from(cmakeBuildDir)
            include("*.dylib", "*.so", "*.dll")
            into(layout.buildDirectory.dir("libs"))
        }
    }
}

tasks.register("build_cmake") {
    description = "Builds native JNI library and copies artifacts"
    group = "build"
    dependsOn(copyNativeLibs)
}
